import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing '{key}'. Available top-level keys: {list(d)}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
_ = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
group_cols = [] if group is None else [group]
protected = set(targets) | set(excluded) | set(group_cols)

def key_like(s):
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return any(t in name for t in ("id", "key", "index"))
    return False

base_feature_columns = [
    c for c in train.columns
    if c not in protected and not key_like(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_7SHSFSVP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    feature_columns = list(preprocessor.feature_names_in_)
    bad = set(feature_columns) & protected
    missing = [c for c in feature_columns if c not in train.columns]
    if bad or missing:
        raise ValueError(f"Preprocessor feature columns conflict: bad={bad}, missing={missing}")
else:
    feature_columns = base_feature_columns

X_raw = train.loc[:, feature_columns]
X_trans = preprocessor.transform(X_raw)

if not isinstance(X_trans, pd.DataFrame):
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    names = list(preprocessor.get_feature_names_out())
    X_trans = pd.DataFrame(X_trans, index=train.index, columns=names)
else:
    X_trans = X_trans.copy()
    X_trans.index = train.index

selector = FunctionTransformer(
    func=None,
    validate=False,
    feature_names_out="one-to-one",
    accept_sparse=False,
)
selector = mlca_runtime.fit_record(selector, X_trans, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")
mlca_runtime.record_metric("n_features", float(X_trans.shape[1]), "train")
print(f"train rows: {X_trans.shape[0]}")
print(f"features preserved: {X_trans.shape[1]}")
