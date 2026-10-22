import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}. Available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded_columns = list(protocol["excluded_columns"])
group_column = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
for col in targets:
    if col not in train.columns:
        raise KeyError(f"Target column {col!r} not found in training table")

excluded = set(excluded_columns) | set(targets)
if group_column is not None:
    excluded.add(group_column)

def key_like(column: str) -> bool:
    s = train[column]
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return bool(s.notna().all() and s.nunique(dropna=False) == len(s))
    if pd.api.types.is_integer_dtype(s.dtype):
        if s.isna().any() or s.nunique(dropna=False) != len(s):
            return False
        values = np.sort(s.to_numpy())
        return len(values) > 1 and bool(np.all(np.diff(values) == 1))
    return False

feature_columns = [
    c for c in train.columns
    if c not in excluded and not key_like(c)
]
if not feature_columns:
    raise ValueError("No feature columns remain after applying protocol exclusions")

X = train.loc[:, feature_columns].copy()
y = train.loc[:, targets].copy()

selector = FunctionTransformer(func=None, validate=False, feature_names_out="one-to-one")
selector = mlca_runtime.fit_record(selector, X, "train", y=y)
selector.n_features_in_ = len(feature_columns)
selector.feature_names_in_ = np.asarray(feature_columns, dtype=object)

mlca_runtime.save_model(selector, "selector_all.pkl")
print(f"train rows: {len(X)}")
print(f"selected features: {len(feature_columns)}")
print("saved selector_all.pkl")
