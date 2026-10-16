import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    n = len(s)
    if s.dtype == object and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n:
        vals = np.sort(s.dropna().unique())
        if len(vals) > 1 and vals[-1] - vals[0] == len(vals) - 1:
            continue
    feature_cols.append(c)

X = train[feature_cols]

selector = FunctionTransformer(validate=False)
mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")

mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(X)), "train")

print(f"features: {len(feature_cols)}")
print(f"train rows: {len(X)}")
print("selector_all.pkl saved")
