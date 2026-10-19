import mlca_runtime, json, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

drop_set = set(targets) | set(excluded)
if group:
    drop_set.add(group)

feature_cols = []
for c in train.columns:
    if c in drop_set:
        continue
    s = train[c]
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

for t in targets:
    if t not in train.columns:
        raise KeyError(f"Target {t} not in training table")

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_input_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            names = get_input_names(step)
            if names is not None:
                return names
    return None

def accepts_raw(obj):
    names = get_input_names(obj)
    if names is None:
        return False
    cols = set(names)
    return (cols.issubset(set(train.columns)) and not (cols & set(targets)) and
            not (cols & set(excluded)) and (group is None or group not in cols))

pre = load_pkl(inp['05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl'])
sel = load_pkl(inp['06_features_retain_features_ZGB5PBQC/selector_all.pkl'])

frontend = None
if accepts_raw(sel):
    frontend = sel
elif accepts_raw(pre):
    pre_out = None
    try:
        pre_out = pre.get_feature_names_out()
    except Exception:
        pass
    sel_in = get_input_names(sel)
    if pre_out is None or sel_in is None or list(pre_out) == list(sel_in):
        frontend = Pipeline([('pre', pre), ('sel', sel)])
    else:
        frontend = pre

if frontend is not None:
    names = get_input_names(frontend)
    if names is not None:
        if set(names) & set(targets) or (set(names) & set(excluded)) or (group and group in names):
            raise ValueError("Saved frontend expects target/excluded/group columns")
        X_train = train[names].copy()
    else:
        X_train = train[feature_cols].copy()
else:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    low_cat = [c for c in cat_cols if train[c].nunique(dropna=False) <= max(1, int(0.05 * len(train)))]
    transformers = []
    if num_cols:
        transformers.append(('num', SimpleImputer(strategy='median'), num_cols))
    if low_cat:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_cat))
    if transformers:
        ct = ColumnTransformer(transformers, remainder='drop')
        width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in low_cat)
        if width <= 300:
            try: ct.set_output(transform='pandas')
            except Exception: pass
        frontend = ct
    else:
        frontend = 'passthrough'
    X_train = train[feature_cols].copy()

y_train = train[targets].copy()
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
steps = ([('frontend', frontend)] if frontend != 'passthrough' else []) + [('model', model)]
pipeline = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("saved alternative_model.pkl")
