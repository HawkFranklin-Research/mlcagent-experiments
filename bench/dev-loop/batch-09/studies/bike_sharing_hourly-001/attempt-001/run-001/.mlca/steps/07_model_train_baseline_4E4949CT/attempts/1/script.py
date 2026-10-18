import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_poisson_deviance

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing {key}; available: {sorted(obj)}")
    return obj[key]

def as_list(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            x = [x]
    return list(x)

protocol = json.loads(paths["_protocol"].read_text())
targets = as_list(need(protocol, "targets"))
excluded = as_list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
task = need(protocol, "task")
metric = need(task, "primary_metric")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

with open(paths["05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_retain_features_R3TMNHH4/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def pipeline_has_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if step is pre or isinstance(step, ColumnTransformer):
            return True
        if "preprocess" in name.lower() or "column" in name.lower():
            return True
    return False

sel_is_frontend = pipeline_has_pre(sel)
frontend = sel if sel_is_frontend else pre

if hasattr(frontend, "feature_names_in_"):
    feature_cols = [str(c) for c in frontend.feature_names_in_]
else:
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and s.is_monotonic_increasing:
            feature_cols.remove(c)

forbidden = set(targets) | set(excluded) | ({group} if group is not None else set())
if forbidden.intersection(feature_cols):
    raise ValueError("Target, excluded, or group columns reached the feature contract")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = train[targets].copy()

if sel_is_frontend:
    model = Pipeline([("frontend", sel), ("model", LinearRegression())])
else:
    steps = [("preprocessor", pre)]
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", LinearRegression()))
    model = Pipeline(steps)

model = mlca_runtime.fit_record(model, X_train, "train", y=y_train)

pred = np.asarray(model.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

if metric == "poisson_deviance":
    vals = []
    for i, target in enumerate(targets):
        yt = val[target].to_numpy()
        yp = np.clip(pred[:, i], 1e-9, None)
        value = float(mean_poisson_deviance(yt, yp))
        mlca_runtime.record_metric(f"{metric}_{target}", value, "val")
        vals.append(value)
    mlca_runtime.record_metric(f"{metric}_mean", float(np.mean(vals)), "val")
    print(f"val {metric} mean: {float(np.mean(vals)):.4f}")

mlca_runtime.save_model(model, "model.pkl")
print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
