import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "targets" not in proto or "task" not in proto:
    raise KeyError(f"Missing required protocol keys; available: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []) or [])
group = proto.get("group_column")
task = proto["task"]
metric = task["primary_metric"]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_QJRR3HF7/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_8C5RPJ11/selector_all.pkl"])
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

base_cols = [c for c in train.columns if c not in targets and c not in excluded and (group is None or c != group)]
keylike = []
for c in base_cols:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and len(s) > 1 and s.nunique(dropna=False) == len(s):
        keylike.append(c)
raw_cols = [c for c in base_cols if c not in keylike]
pre_cols = getattr(pre, "feature_names_in_", None)
if pre_cols is not None:
    pre_cols = [str(c) for c in pre_cols]
    raw_cols = [c for c in pre_cols if c in base_cols and c not in keylike]
if not raw_cols:
    raw_cols = [c for c in base_cols if c not in keylike]

def names(obj):
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(c) for c in step.feature_names_in_]
    return None

def accepts_raw(obj, cols):
    n = names(obj)
    if n is not None and n == list(cols):
        return True
    if isinstance(obj, Pipeline):
        first = next(iter(obj.named_steps.values()))
        n2 = names(first)
        if n2 is not None and n2 == list(cols):
            return True
    return False

X_train = train[raw_cols].copy()
y_train = train[targets].copy()
X_val = val[raw_cols].copy()
y_val = val[targets].copy()

if accepts_raw(sel, raw_cols):
    model = Pipeline([("frontend", sel), ("model", LinearRegression())])
else:
    if pre is None:
        raise ValueError("Selector does not accept raw features and no preprocessor is available")
    model = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("model", LinearRegression()),
    ])

model = mlca_runtime.fit_record(model, X_train, "train", y=y_train)

pred = np.asarray(model.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(y_val[t], pred[:, i])))
    mlca_runtime.record_metric(f"rmse_{t}", rmse, "val")
    rmses.append(rmse)
headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
mlca_runtime.save_model(model, "model.pkl")
print(f"features={len(raw_cols)} targets={len(targets)}")
print(f"val_{metric}={headline:.4f}")
