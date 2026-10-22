import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d)}")
    return d[k]

targets = list(need(proto, "targets"))
excluded = list(need(proto, "excluded_columns"))
group = proto.get("group_column")
task = need(proto, "task")
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task)}")
metric = task["primary_metric"]

exclusions = set(targets) | set(excluded)
if group is not None:
    exclusions.add(group)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
for mk in ["05_preprocess_fit_preprocessing_5ED3RGCF/manifest.json",
           "06_features_retain_features_YGN0TN24/manifest.json"]:
    json.loads(inp[mk].read_text())

with open(inp["05_preprocess_fit_preprocessing_5ED3RGCF/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_YGN0TN24/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline) and obj.steps:
        return input_cols(obj.steps[0][1])
    return None

def out_cols(obj):
    try:
        return set(map(str, obj.get_feature_names_out()))
    except Exception:
        return None

pre_cols = input_cols(pre)
if pre_cols:
    raw_cols = [c for c in pre_cols if c in train.columns and c not in exclusions]
else:
    raw_cols = [c for c in train.columns if c not in exclusions]
if not raw_cols:
    raise ValueError("No feature columns remain after applying contract")

sel_in = input_cols(sel)
pre_out = out_cols(pre)
if sel_in is not None and pre_out is not None and set(sel_in) == pre_out:
    features = Pipeline([("pre", pre), ("sel", sel)])
elif sel_in is not None and set(sel_in) == set(raw_cols):
    features = sel
else:
    features = pre

model = Pipeline([("features", features), ("model", LinearRegression())])
mlca_runtime.fit_record(model, train[raw_cols], "train", y=train[targets])
mlca_runtime.save_model(model, "model.pkl")

pred = np.asarray(model.predict(val[raw_cols]))
if pred.ndim == 1:
    pred = pred[:, None]
errs = []
for j, t in enumerate(targets):
    e = val[t].to_numpy() - pred[:, j]
    rmse = float(np.sqrt(np.mean(e * e)))
    errs.append(rmse)
    mlca_runtime.record_metric(f"{metric}_{t}", rmse, "val")
mean_rmse = float(np.mean(errs))
mlca_runtime.record_metric(f"{metric}_mean", mean_rmse, "val")
print(f"train={len(train)} features={len(raw_cols)}")
print(f"val {metric} mean={mean_rmse:.4g}")
