import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = protocol.get("group_column")
task = need(protocol, "task")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_T5EPBF1H/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_EWX96D8N/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

raw_cols = set(train.columns)
bad = set(targets) | set(excluded) | ({group} if group else set())
cands = []
for name, obj in [("pre", pre), ("sel", sel)]:
    cols = input_cols(obj)
    if cols and set(cols).issubset(raw_cols) and not (set(cols) & bad):
        cands.append((name, obj, cols))
if not cands:
    raise RuntimeError("No saved frontend accepts the raw feature columns")
name, frontend, feature_cols = next((c for c in cands if c[0] == "pre"), cands[0])

X_train = train[feature_cols]
y_train = train[targets]
X_val = val[feature_cols]
y_val = val[targets].to_numpy()

pipe = Pipeline([
    ("preprocess", frontend),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])
fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

pred = fitted.predict(X_val)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, i] - y_val[:, i]) ** 2)))
    rmses.append(rmse)
    mlca_runtime.record_metric(f"rmse_{t}", rmse, "val")
mean_rmse = float(np.mean(rmses))
mlca_runtime.record_metric(primary_metric, mean_rmse, "val")
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"features={len(feature_cols)} train={len(X_train)} val={len(X_val)}")
print(f"val_{primary_metric}={mean_rmse:.4f}")
