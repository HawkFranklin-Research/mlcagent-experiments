import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]
for c in list(feat_cols):
    s = train[c]
    if s.nunique() == len(train) and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
        feat_cols.remove(c)

with open(inputs["05_preprocess_fit_preprocessing_ZR2ARXZ1/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_H6GS4FCS/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def get_input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            cols = get_input_cols(step)
            if cols:
                return cols
    return None

sel_cols = get_input_cols(sel)
if sel_cols is not None and all(c in feat_cols for c in sel_cols):
    pipe = Pipeline([("frontend", sel), ("model", LinearRegression())])
else:
    pipe = Pipeline([("pre", pre), ("sel", sel), ("model", LinearRegression())])

X = train[feat_cols]
y = train[targets]
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

Xv = val[feat_cols]
yv = val[targets]
pred = fitted.predict(Xv)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
pred = np.clip(pred, 1e-6, None)
devs = []
for i, t in enumerate(targets):
    d = float(mean_poisson_deviance(yv[t].to_numpy(), pred[:, i]))
    mlca_runtime.record_metric(f"poisson_deviance_{t}", d, "val")
    devs.append(d)
mlca_runtime.record_metric("poisson_deviance_mean", float(np.mean(devs)), "val")

print("train rows:", len(train), "val rows:", len(val))
print("features:", len(feat_cols), "targets:", targets)
print("val poisson_deviance mean:", round(float(np.mean(devs)), 4))
