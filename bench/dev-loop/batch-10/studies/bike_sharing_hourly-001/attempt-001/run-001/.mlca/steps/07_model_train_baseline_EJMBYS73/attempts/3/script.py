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

def need(k):
    if k not in protocol:
        raise KeyError(f"missing {k!r}; available: {list(protocol)}")
    return protocol[k]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
group = protocol.get("group_column")
task = need("task")
metric = task.get("primary_metric", "rmse")
if not targets:
    raise ValueError("no targets declared")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"]).reindex(columns=train.columns)

excl = set(targets) | set(excluded)
if group is not None:
    excl.add(group)

def is_key_like(s):
    n = len(s)
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n:
        u = np.unique(s.dropna().to_numpy())
        if len(u) > 0 and np.array_equal(u, np.arange(u.min(), u.min() + len(u))):
            return True
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == n:
        return True
    return False

feature_cols = [c for c in train.columns if c not in excl and not is_key_like(train[c])]
for c in list(targets) + list(excluded):
    if c in feature_cols:
        raise ValueError(f"excluded column {c!r} in feature set")

with open(inputs["05_preprocess_fit_preprocessing_R4WEQED5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_EN84T9CF/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def set_pd(o):
    if hasattr(o, "set_output"):
        try:
            o.set_output(transform="pandas")
        except Exception:
            pass
    return o

pre = set_pd(pre)
sel = set_pd(sel)

def inames(o):
    if o is None:
        return None
    if hasattr(o, "feature_names_in_"):
        try:
            return list(o.feature_names_in_)
        except Exception:
            return None
    steps = getattr(o, "steps", None)
    if steps is not None:
        for _, s in steps:
            r = inames(s)
            if r is not None:
                return r
    return None

def raw_ok(names):
    if not names:
        return False
    if not all(c in feature_cols for c in names):
        return False
    if any(c in excl for c in names):
        return False
    return True

pre_in = inames(pre)
sel_in = inames(sel)

ttr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)

if raw_ok(sel_in):
    cols = list(sel_in)
    pipe = Pipeline([("features", sel), ("model", ttr)])
elif raw_ok(pre_in):
    cols = list(pre_in)
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", ttr)])
else:
    cols = list(feature_cols)
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", ttr)])

Xtr = train[cols].reindex(columns=cols)
ytr = train[targets]
fitted = mlca_runtime.fit_record(pipe, Xtr, "train", ytr)

Xva = val.reindex(columns=cols)
try:
    pred = np.asarray(fitted.predict(Xva))
except Exception as e:
    missing = [c for c in cols if c not in Xva.columns]
    raise ValueError(f"predict failed ({type(e).__name__}: {str(e)[:200]}); missing={missing}") from e

if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[0] != len(val):
    raise ValueError(f"prediction length {pred.shape[0]} != val length {len(val)}")

rmses = []
for i, t in enumerate(targets):
    y_true = val[t].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, pred[:, i])))
    mlca_runtime.record_metric(f"rmse_{t}", rmse, "val")
    rmses.append(rmse)
headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train={train.shape} val={val.shape} features={len(cols)}")
print(f"val {metric}={headline:.4f}")
