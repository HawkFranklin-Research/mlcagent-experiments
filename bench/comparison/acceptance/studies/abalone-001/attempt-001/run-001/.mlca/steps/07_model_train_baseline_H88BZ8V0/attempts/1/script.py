import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
task = need(protocol, "task")
primary = need(task, "primary_metric")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

with open(inp["05_preprocess_fit_preprocessing_QTKFYNMD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_40Q8191Y/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    if obj is None:
        return None
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None

def accepts_raw(obj, raw_cols):
    names = input_names(obj)
    return names is not None and list(names) == list(raw_cols)

pre_names = input_names(pre)
if pre_names is not None:
    feature_cols = [c for c in pre_names if c in train.columns and c not in targets and c not in excluded and c != group]
else:
    def key_like(s):
        if s.dtype.kind == "f":
            return False
        if s.dtype.kind in "ifuO":
            return len(s) > 1 and s.nunique(dropna=False) == len(s)
        return False
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and not key_like(train[c])]

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
X_val = val[feature_cols].copy()
y_val = val[targets].to_numpy()

if sel is not None and accepts_raw(sel, feature_cols):
    pipe = Pipeline([("selector", sel), ("model", LinearRegression())])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", LinearRegression())])

model = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
pred = np.asarray(model.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

rmses = []
for i, t in enumerate(targets):
    v = float(np.sqrt(mean_squared_error(y_val[:, i], pred[:, i])))
    rmses.append(v)
    if len(targets) > 1:
        mlca_runtime.record_metric(f"{primary}_{t}", v, "val")
headline = float(np.mean(rmses))
mlca_runtime.record_metric(primary, headline, "val")
mlca_runtime.save_model(model, "model.pkl")
print(f"features={len(feature_cols)} train={X_train.shape} {primary}={headline:.4g}")
