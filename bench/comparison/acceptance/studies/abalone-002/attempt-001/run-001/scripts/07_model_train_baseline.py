import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
metric = (proto.get("task") or {}).get("primary_metric") or "rmse"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_2K81ED4E/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)


def has_any(p, names):
    return isinstance(p, Pipeline) and any(n in p.named_steps for n in names)


def contains(p, obj):
    return isinstance(p, Pipeline) and any(s is obj for s in p.named_steps.values())


def cols_of(o):
    c = getattr(o, "feature_names_in_", None)
    return list(c) if c is not None else []


sel_names = ("selector", "selector_all", "select")
pre_names = ("preprocess", "preprocessor", "preprocessing")
if contains(sel, pre) or has_any(sel, pre_names):
    frontend, need_sel = sel, False
elif contains(pre, sel) or has_any(pre, sel_names):
    frontend, need_sel = pre, False
else:
    frontend, need_sel = pre, True

feature_cols = cols_of(frontend) or cols_of(pre) or cols_of(sel)
if not feature_cols:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
feature_cols = [c for c in feature_cols if c in train.columns]

X_train = train.loc[:, feature_cols]
y_train = train.loc[:, targets]
X_val = val.loc[:, feature_cols]
y_val = val.loc[:, targets]

steps = [("pre", frontend)]
if need_sel:
    steps.append(("sel", sel))
steps.append(("model", LinearRegression()))
pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "model.pkl")

pred = np.asarray(pipe.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
rmses = []
for i, t in enumerate(targets):
    d = pred[:, i] - y_val.iloc[:, i].to_numpy()
    r = float(np.sqrt(np.mean(d ** 2)))
    rmses.append(r)
    mlca_runtime.record_metric(metric + "_" + str(t), r, "val")
headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
print(f"train={train.shape} val={val.shape} features={len(feature_cols)} {metric}={headline:.4g}")
