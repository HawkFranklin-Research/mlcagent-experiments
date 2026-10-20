import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available keys: {sorted(d.keys())}")
    return d[key]


targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
primary_metric = need(need(protocol, "task"), "primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

target_set, excluded_set = set(targets), set(excluded)
group_name = group if group is not None else None

if hasattr(pre, "feature_names_in_"):
    feature_cols = [str(c) for c in pre.feature_names_in_]
    bad = set(feature_cols) & (target_set | excluded_set)
    if bad:
        raise ValueError(f"Preprocessor includes target/excluded columns: {sorted(bad)}")
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor requires missing columns: {missing}")
else:
    n = len(train)
    feature_cols = []
    for c in train.columns:
        if c in target_set or c in excluded_set or c == group_name:
            continue
        s = train[c]
        is_text = (
            pd.api.types.is_object_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s.dtype)
        )
        if is_text and s.nunique(dropna=True) == n:
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == n and (
            s.is_monotonic_increasing or s.is_monotonic_decreasing
        ):
            continue
        feature_cols.append(c)

est = Pipeline([
    ("preprocessor", pre),
    ("impute", SimpleImputer()),
    ("model", LinearRegression()),
])

X_train = train[feature_cols].copy()
y_train = train[list(targets)].copy()
fitted = mlca_runtime.fit_record(est, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

if val is not None and len(val):
    X_val = val[feature_cols].copy()
    y_val = val[list(targets)].to_numpy()
    pred = np.asarray(fitted.predict(X_val))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    rmses = []
    for j, name in enumerate(targets):
        rmse = float(np.sqrt(np.mean((pred[:, j] - y_val[:, j]) ** 2)))
        rmses.append(rmse)
        mlca_runtime.record_metric(f"{primary_metric}_{name}", rmse, "val")
    headline = float(np.mean(rmses))
    mlca_runtime.record_metric(primary_metric, headline, "val")
else:
    headline = float("nan")

print(f"train rows={len(train)} features={len(feature_cols)} targets={len(targets)}")
print(f"val rows={len(val)} {primary_metric}={headline:.4f}")
