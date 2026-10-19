import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())

def req(obj, key, name):
    if key not in obj:
        raise KeyError(f"{name} missing {key!r}; available keys: {sorted(obj)}")
    return obj[key]

targets = list(req(protocol, "targets", "_protocol"))
excluded = list(req(protocol, "excluded_columns", "_protocol"))
group_col = req(protocol, "group_column", "_protocol")
task = req(protocol, "task", "_protocol")
primary_metric = req(task, "primary_metric", "_protocol['task']") or "rmse"

train = pd.read_parquet(ins["train"])
val = pd.read_parquet(ins["val"])
for t in targets:
    if t not in train.columns or t not in val.columns:
        raise KeyError(f"Target {t!r} missing from data")

def key_like(s):
    n = int(s.notna().sum())
    if n <= 1:
        return False
    if pd.api.types.is_integer_dtype(s):
        if set(s.dropna().unique()) <= {0, 1}:
            return False
        return s.nunique(dropna=True) == n
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    return s.nunique(dropna=True) == n

drop = set(targets) | set(excluded)
if group_col not in (None, ""):
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain")

num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) <= max(2, int(0.05 * len(train))):
        cat_cols.append(c)
if date_cols:
    raise ValueError(f"Datetime columns not supported: {date_cols}")

transformers = []
if num_cols:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))
if not transformers:
    raise ValueError("No supported feature columns remain")

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")
model = Pipeline([("preprocessor", pre), ("regressor", LinearRegression())])

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

X_val = val[feature_cols].copy()
pred = np.asarray(fitted.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, i] - val[t].to_numpy()) ** 2)))
    rmses.append(rmse)
    mlca_runtime.record_metric(f"{primary_metric}_{t}", rmse, "val")
headline = float(np.mean(rmses))
mlca_runtime.record_metric(primary_metric, headline, "val")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"val {primary_metric} mean={headline:.4f}")
