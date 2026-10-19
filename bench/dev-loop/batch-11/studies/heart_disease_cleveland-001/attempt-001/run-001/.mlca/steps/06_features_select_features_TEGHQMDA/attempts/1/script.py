import json, pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target = targets[0]

drop = set(targets) | set(excluded) | ({group} if group else set())

def is_keylike(s):
    if s.name in drop:
        return True
    name = str(s.name).lower()
    if 'unnamed' in name or name in ('id', 'index', 'row_id', 'patient_id'):
        return True
    n = len(s)
    if n and s.nunique(dropna=False) == n:
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            return True
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return True
    return False

feature_cols = [c for c in train.columns if not is_keylike(train[c])]
print(f"n_features: {len(feature_cols)}")

with open(inputs['05_preprocess_fit_preprocessing_69FZY9R2/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
print(f"pre: {type(pre).__name__}", list(getattr(pre, 'named_steps', {}).keys()))

pre_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else None
if pre_cols is not None:
    pre_cols = [c for c in pre_cols if c in feature_cols] or feature_cols

already_select = False
if isinstance(pre, Pipeline):
    if any(n in ('select', 'selector', 'selectkbest') for n in pre.named_steps):
        already_select = True

if already_select:
    pipe = pre
    fit_cols = pre_cols if pre_cols else feature_cols
else:
    n_feat = len(pre_cols) if pre_cols else len(feature_cols)
    k = min(20, n_feat)
    pipe = Pipeline([
        ('pre', pre),
        ('select', SelectKBest(mutual_info_classif, k=k, random_state=42)),
    ])
    fit_cols = pre_cols if pre_cols else feature_cols

try:
    width = len(pre.get_feature_names_out())
except Exception:
    width = None
if width is not None and width <= 300:
    try:
        pipe.set_output(transform='pandas')
    except Exception:
        pass

X = train[fit_cols]
y = train[target]
mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'selector.pkl')
print("saved selector.pkl")
