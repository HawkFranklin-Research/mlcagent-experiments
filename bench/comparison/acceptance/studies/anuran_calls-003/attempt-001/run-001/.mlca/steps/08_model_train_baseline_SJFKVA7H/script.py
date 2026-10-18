import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = req(protocol, 'targets')
excluded = req(protocol, 'excluded_columns')
group = req(protocol, 'group_column')
_ = req(protocol, 'task')['primary_metric']

train = pd.read_parquet(inp['train'])

def load_pkl(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

pre = load_pkl('06_preprocess_fit_preprocessing_D9KP7ADR/preprocessor.pkl')
sel = load_pkl('07_features_retain_features_11DAHWEE/selector_all.pkl')

def feat_names(obj):
    return list(getattr(obj, 'feature_names_in_', [])) if obj is not None else []

pre_in, sel_in = feat_names(pre), feat_names(sel)

def contains_pre(pipe, pre, pre_in):
    if not isinstance(pipe, Pipeline):
        return False
    for step in pipe.named_steps.values():
        if step is pre:
            return True
        if isinstance(step, (Pipeline, ColumnTransformer)):
            if pre_in and list(getattr(step, 'feature_names_in_', [])) == pre_in:
                return True
            if contains_pre(step, pre, pre_in):
                return True
    return False

def raw_feature_cols():
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    cols = [c for c in train.columns if c not in drop]
    keep = []
    for c in cols:
        s = train[c]
        is_text = (
            pd.api.types.is_object_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )
        if is_text and s.notna().all() and s.nunique(dropna=True) == len(s):
            continue
        if (
            pd.api.types.is_integer_dtype(s.dtype)
            and not pd.api.types.is_bool_dtype(s.dtype)
            and s.notna().all()
            and s.nunique(dropna=True) == len(s)
        ):
            continue
        keep.append(c)
    return keep

if pre is not None and sel is not None and contains_pre(sel, pre, pre_in):
    steps = [('frontend', sel)]
    raw_cols = sel_in or pre_in or raw_feature_cols()
elif pre is not None and sel is not None:
    steps = [('pre', pre), ('sel', sel)]
    raw_cols = pre_in or raw_feature_cols()
elif pre is not None:
    steps = [('pre', pre)]
    raw_cols = pre_in or raw_feature_cols()
elif sel is not None:
    steps = [('sel', sel)]
    raw_cols = sel_in or raw_feature_cols()
else:
    steps = []
    raw_cols = raw_feature_cols()

bad = set(raw_cols) & (set(targets) | set(excluded) | ({group} if group is not None else set()))
if bad:
    raise ValueError(f"Feature list includes target/excluded/group: {sorted(bad)}")

X = train[raw_cols].copy()
y = train[targets].copy()
clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
pipeline = Pipeline(steps + [('clf', clf)])
model = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

print(f"fit rows={len(X)} features={len(raw_cols)} targets={len(targets)}")
print("saved model.pkl")
