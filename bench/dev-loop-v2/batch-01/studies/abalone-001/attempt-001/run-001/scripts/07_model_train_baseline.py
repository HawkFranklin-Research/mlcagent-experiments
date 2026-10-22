import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {sorted(d)}")
    return d[k]

targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group_col = req(protocol, "group_column")
task = req(protocol, "task")
metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return len(s) > 1 and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        v = s.dropna().to_numpy()
        return len(v) == len(s) and len(v) > 1 and np.array_equal(np.sort(v), np.arange(v.min(), v.max() + 1))
    return False

blocked = set(targets) | set(excluded)
if group_col:
    blocked.add(group_col)
feature_cols = [c for c in train.columns if c not in blocked and not key_like(train[c])]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pre = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols),
], remainder="drop")
pre.set_output(transform="pandas")

model = Pipeline([
    ("preprocess", pre),
    ("regressor", LinearRegression()),
])
model.set_output(transform="pandas")

X_train = train[feature_cols]
y_train = train[targets]
fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

X_val = val[feature_cols]
y_val = val[targets].to_numpy()
pred = np.asarray(fitted.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if y_val.ndim == 1:
    y_val = y_val.reshape(-1, 1)
rmse = np.sqrt(np.mean((pred - y_val) ** 2, axis=0))
for t, v in zip(targets, rmse):
    mlca_runtime.record_metric(f"{metric}_{t}", float(v), "val")
mlca_runtime.record_metric(metric, float(np.mean(rmse)), "val")

print("train", train.shape, "features", len(feature_cols))
print("val", val.shape)
print("val", metric, f"{float(np.mean(rmse)):.4g}")
