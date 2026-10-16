import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)):
        if s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

X_train = train[feature_cols]
y_train = train[targets]

with open(inputs["05_preprocess_fit_preprocessing_VM7ERF98/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_RGQ2HW1B/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def wraps_pre(obj):
    return isinstance(obj, Pipeline) and len(obj.named_steps) >= 2

if wraps_pre(sel):
    frontend = sel
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

model = Pipeline([
    ("frontend", frontend),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

X_val = val[feature_cols]
y_val = val[targets]
pred = np.asarray(fitted.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

per_target = {}
for i, t in enumerate(targets):
    per_target[t] = float(np.sqrt(np.mean((pred[:, i] - y_val[t].values) ** 2)))
headline = float(np.mean(list(per_target.values())))
mlca_runtime.record_metric(primary_metric, headline, "val")
for t, v in per_target.items():
    mlca_runtime.record_metric(f"{primary_metric}_{t}", v, "val")

print(f"features: {len(feature_cols)}")
print(f"train: {X_train.shape}, val: {X_val.shape}")
print(f"{primary_metric}: {headline:.4f}")
