import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"missing {key!r}; available top-level keys: {list(protocol)}")
    return protocol[key]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
group = protocol.get("group_column")
task = need("task")
metric = task.get("primary_metric", "rmse")
if not targets:
    raise ValueError("no targets declared")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
excl = set(targets) | set(excluded)
if group is not None:
    excl.add(group)

def is_key_like(s):
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        u = np.unique(s.dropna().to_numpy())
        if len(u) and np.array_equal(u, np.arange(u.min(), u.min() + len(u))):
            return True
    return False

feature_cols = [c for c in train.columns if c not in excl and not is_key_like(train[c])]
for c in targets + excluded:
    if c in feature_cols:
        raise ValueError(f"excluded column {c!r} reached feature contract")

with open(inputs["05_preprocess_fit_preprocessing_R4WEQED5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_EN84T9CF/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def first_input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            names = first_input_names(step)
            if names is not None:
                return names
    return None

pre_in = first_input_names(pre)
sel_in = first_input_names(sel)

def usable_raw(names):
    return names is not None and all(c in feature_cols for c in names) and not any(c in excl for c in names)

ttr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
if usable_raw(sel_in):
    cols = list(sel_in)
    steps = [("features", sel), ("model", ttr)]
elif usable_raw(pre_in):
    cols = list(pre_in)
    steps = [("preprocess", pre), ("select", sel), ("model", ttr)]
else:
    cols = list(feature_cols)
    steps = [("preprocess", pre), ("select", sel), ("model", ttr)]

pipeline = Pipeline(steps)
X = train[cols]
y = train[targets]
fitted = mlca_runtime.fit_record(pipeline, X, "train", y)

pred = np.asarray(fitted.predict(val[cols]))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(val[t].to_numpy(), pred[:, i])))
    mlca_runtime.record_metric(f"rmse_{t}", rmse, "val")
    rmses.append(rmse)
headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train={train.shape} val={val.shape} features={len(cols)}")
print(f"val {metric}={headline:.4f}")
