import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing key {key!r}. Available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = set(need(protocol, "targets"))
excluded = set(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")

train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == n:
        return s.is_monotonic_increasing or s.is_monotonic_decreasing
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets
    and c not in excluded
    and (group is None or c != group)
    and not is_key_like(train[c])
]

X = train[feature_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_1E8V578C/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    if set(expected) == set(feature_cols):
        X = X[expected]
    else:
        raise ValueError("preprocessor feature_names_in_ does not match training feature columns")

X_trans = pre.transform(X)
if not isinstance(X_trans, pd.DataFrame):
    names = pre.get_feature_names_out(X.columns) if hasattr(pre, "get_feature_names_out") else feature_cols
    X_trans = pd.DataFrame(X_trans, columns=names, index=X.index)

selector = Pipeline([("passthrough", "passthrough")])
selector = mlca_runtime.fit_record(selector, X_trans, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

if hasattr(selector, "feature_names_in_"):
    assert list(selector.feature_names_in_) == list(X_trans.columns)

print(f"feature columns: {len(feature_cols)}")
print(f"transformed shape: {X_trans.shape}")
print("saved selector_all.pkl")
