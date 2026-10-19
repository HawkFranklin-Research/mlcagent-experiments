import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
needed = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
_ = protocol["task"]

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]

protected = set(targets) | set(excluded) | {group}
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
_ = json.loads(inputs["05_preprocess_fit_preprocessing_WHDXCN7B/manifest.json"].read_text())

def is_key_like(s):
    if s.name in protected:
        return False
    n = s.notna().sum()
    if n < 2 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        name = str(s.name).lower()
        return (
            any(tok in name for tok in ("id", "key", "code", "index", "record", "seq"))
            or s.is_monotonic_increasing
            or s.is_monotonic_decreasing
        )
    return False

feature_cols = [
    c for c in train.columns
    if c not in protected and not is_key_like(train[c])
]

if hasattr(pre, "feature_names_in_"):
    pre_cols = [str(c) for c in pre.feature_names_in_]
    if set(pre_cols).issubset(set(feature_cols)):
        feature_cols = pre_cols

assert not (set(feature_cols) & protected)
X = train.loc[:, feature_cols].copy()

selector = ColumnTransformer(transformers=[], remainder="passthrough")
selector.set_output(transform="pandas")
selector = mlca_runtime.fit_record(selector, X, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train rows={len(train)} features={len(feature_cols)}")
