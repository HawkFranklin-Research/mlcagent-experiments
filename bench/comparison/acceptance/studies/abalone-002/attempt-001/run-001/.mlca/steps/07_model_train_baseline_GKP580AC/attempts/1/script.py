import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in proto:
        raise KeyError(f"Missing _protocol key {key!r}; available: {list(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto["group_column"]
task = proto["task"]
metric = task.get("primary_metric") or "rmse"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
pre_manifest = json.loads(inp["05_preprocess_fit_preprocessing_FG7YG4JM/manifest.json"].read_text())
sel_manifest = json.loads(inp["06_features_retain_features_2K81ED4E/manifest.json"].read_text())
with open(inp["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_2K81ED4E/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)


def contains(pipe, obj):
    return isinstance(pipe, Pipeline) and any(step is obj for step in pipe.named_steps.values())


def has_name(pipe, names):
    return isinstance(pipe, Pipeline) and any(n in pipe.named_steps for n in names)


def key_like(s):
    return len(s) == s.nunique(dropna=True) and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )


def manifest_cols(m):
    for k in ("feature_columns", "features", "input_columns", "input_features", "columns"):
        v = m.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return list(v)
    return None


for obj in (pre, sel):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

feature_cols = getattr(pre, "feature_names_in_", None)
if feature_cols is None and (contains(sel, pre) or has_name(sel, ("preprocess", "preprocessor", "preprocessing"))):
    feature_cols = getattr(sel, "feature_names_in_", None)
if feature_cols is None:
    feature_cols = manifest_cols(pre_manifest)
if feature_cols is None:
    feature_cols = manifest_cols(sel_manifest)
if feature_cols is None:
    feature_cols = [
        c for c in train.columns
        if c not in set(targets + excluded + [group]) and not key_like(train[c])
    ]

stop = set(targets + excluded + [group])
feature_cols = [
    c for c in feature_cols
    if c in train.columns and c not in stop and not key_like(train[c])
]
if not feature_cols:
    raise ValueError("No usable feature columns after applying the feature contract")

X_train = train.loc[:, feature_cols]
y_train = train.loc[:, targets]

if contains(pre, sel) or has_name(pre, ("selector", "select", "selector_all")):
    frontend, need_sel = pre, False
elif contains(sel, pre) or has_name(sel, ("preprocess", "preprocessor", "preprocessing")):
    frontend, need_sel = sel, False
else:
    frontend, need_sel = pre, True

keep = ColumnTransformer([("keep", "passthrough", feature_cols)], remainder="drop")
try:
    keep.set_output(transform="pandas")
except Exception:
    pass

steps = [("features", keep), ("frontend", frontend)]
if need_sel:
    steps.append(("selector", sel))
steps.append(("model", LinearRegression()))
pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "model.pkl")

X_val = val.loc[:, feature_cols]
y_val = val.loc[:, targets]
pred = np.asarray(pipe.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

rmses = []
for i, target in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, i] - y_val.iloc[:, i].to_numpy()) ** 2)))
    rmses.append(rmse)
    mlca_runtime.record_metric(f"{metric}_{target}", rmse, "val")

headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
print(f"train={train.shape} val={val.shape} features={len(feature_cols)} {metric}={headline:.4g}")
