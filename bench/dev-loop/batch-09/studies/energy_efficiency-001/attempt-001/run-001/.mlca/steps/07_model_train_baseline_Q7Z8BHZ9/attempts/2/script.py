import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(obj)}")
    return obj[key]

targets = list(need(protocol, "targets"))
excluded = set(need(protocol, "excluded_columns"))
task = need(protocol, "task")
primary = need(task, "primary_metric")
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

for col in targets:
    if col not in train.columns or col not in val.columns:
        raise KeyError(f"Target {col!r} missing; train columns: {list(train.columns)}")

def is_key_like(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_integer_dtype(s)

target_set = set(targets)
feature_cols = [
    c for c in train.columns
    if c not in target_set and c not in excluded and c != group and not is_key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after contract exclusions")

pre = ColumnTransformer([("features", "passthrough", feature_cols)], remainder="drop")
pre.set_output(transform="pandas")
model = Pipeline([("pre", pre), ("model", LinearRegression())])

X_train = train[feature_cols]
y_train = train[targets]
fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)

pred = np.asarray(fitted.predict(val[feature_cols]))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
y_val = val[targets].to_numpy()
rmses = np.sqrt(np.mean((pred - y_val) ** 2, axis=0))
for target, value in zip(targets, rmses):
    mlca_runtime.record_metric(f"{primary}_{target}", float(value), "val")
mlca_runtime.record_metric(f"{primary}_mean", float(np.mean(rmses)), "val")

mlca_runtime.save_model(fitted, "model.pkl")
print("features:", len(feature_cols))
print("train/val:", train.shape, val.shape)
print("rmse:", ", ".join(f"{t}={v:.4f}" for t, v in zip(targets, rmses)))
print("rmse_mean:", f"{float(np.mean(rmses)):.4f}")
