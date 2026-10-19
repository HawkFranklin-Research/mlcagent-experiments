import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])


def keylike(s, name):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        n = name.lower()
        return any(t in n for t in ("id", "key", "code"))
    return False


drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c], c)]
if not feature_cols:
    raise ValueError("No feature columns available")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy() if val is not None else None


def load_pkl(key):
    with open(paths[key], "rb") as f:
        return pickle.load(f)


baseline = load_pkl("07_model_train_baseline_Q9V0XT19/model.pkl")
pre = load_pkl("05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_E2FAAE9F/selector_all.pkl")


def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return None


def accepts_raw(obj):
    names = input_names(obj)
    return names is not None and names == feature_cols


frontend = None
if isinstance(baseline, Pipeline) and len(baseline.steps) > 1 and accepts_raw(baseline):
    frontend = Pipeline(baseline.steps[:-1])
elif accepts_raw(sel):
    frontend = sel
elif accepts_raw(pre):
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)]) if sel is not None else pre
elif isinstance(pre, Pipeline) and accepts_raw(pre):
    frontend = pre
else:
    raise ValueError("No saved frontend accepts the raw feature columns")


def dense(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return np.asarray(a)


def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 2 and b.shape[1] == 1:
        a = a.reshape(-1, 1)
    return float(np.mean((a - b) ** 2))


y_train = dense(frontend.transform(X_train))
mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
steps = list(frontend.steps) if isinstance(frontend, Pipeline) else [("frontend", frontend)]
model = Pipeline(steps + [("mlp", mlp)])

fitted = mlca_runtime.fit_record(model, X_train, partition="train", y=y_train)

mse_train = mse(fitted.predict(X_train), y_train)
mlca_runtime.record_metric(metric, mse_train, "train")

mse_val = None
if X_val is not None and len(X_val):
    front_fitted = fitted[:-1]
    y_val = dense(front_fitted.transform(X_val))
    mse_val = mse(fitted.predict(X_val), y_val)
    mlca_runtime.record_metric(metric, mse_val, "val")

mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"train rows={len(X_train)} features={len(feature_cols)}")
print(f"processed_dim={y_train.shape[1]}")
print(f"train {metric}={mse_train:.4g}")
if mse_val is not None:
    print(f"val {metric}={mse_val:.4g}")
