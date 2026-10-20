import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
task = need(protocol, "task")
primary_metric = need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_W9C3S8F0/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

all_cols = list(train.columns)
target_set, excluded_set = set(targets), set(excluded)
cand = [c for c in all_cols if c not in target_set and c not in excluded_set and c != group]
n = len(train)


def is_text_dtype(s):
    return (
        pd.api.types.is_object_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(s.dtype)
    )


def key_like(c):
    s = train[c]
    if is_text_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=True) == n and (
            s.is_monotonic_increasing or s.is_monotonic_decreasing
        )
    return False


feature_cols = [c for c in cand if not key_like(c)]
if not feature_cols:
    raise ValueError("No feature columns remain")


def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return None


def output_names(obj):
    try:
        return [str(x) for x in obj.get_feature_names_out()]
    except Exception:
        return None


pre_in = input_names(pre)
if pre_in is not None:
    pre_in_str = [str(x) for x in pre_in]
    bad = set(pre_in_str) & (
        set(map(str, targets))
        | set(map(str, excluded))
        | ({str(group)} if group is not None else set())
    )
    if bad:
        raise ValueError(f"Saved preprocessor includes excluded/target columns: {sorted(bad)}")
    cand_map = {str(c): c for c in cand}
    if all(x in cand_map for x in pre_in_str):
        feature_cols = [cand_map[x] for x in pre_in_str]
feature_cols_str = [str(c) for c in feature_cols]

sel_in = input_names(sel)
pre_out = output_names(pre)
sel_in_str = [str(x) for x in sel_in] if sel_in is not None else None
pre_in_str = [str(x) for x in pre_in] if pre_in is not None else None

if (
    sel_in_str is not None
    and len(sel_in_str) == len(feature_cols_str)
    and set(sel_in_str) == set(feature_cols_str)
):
    est = Pipeline([("frontend", sel), ("model", LinearRegression())])
elif (
    pre_in_str is not None
    and len(pre_in_str) == len(feature_cols_str)
    and set(pre_in_str) == set(feature_cols_str)
    and pre_out is not None
    and sel_in_str == pre_out
):
    est = Pipeline(
        [("preprocessor", pre), ("selector", sel), ("model", LinearRegression())]
    )
elif (
    pre_in_str is not None
    and len(pre_in_str) == len(feature_cols_str)
    and set(pre_in_str) == set(feature_cols_str)
):
    est = Pipeline([("preprocessor", pre), ("model", LinearRegression())])
else:
    numeric = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(train[c])
        and not pd.api.types.is_datetime64_any_dtype(train[c])
    ]
    cat = [
        c
        for c in feature_cols
        if c not in numeric and is_text_dtype(train[c])
        and train[c].nunique(dropna=True) <= max(20, int(0.05 * n))
    ]
    transformers = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
        )
    ct = ColumnTransformer(transformers, remainder="drop")
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass
    est = Pipeline([("preprocessor", ct), ("model", LinearRegression())])

if "frontend" in est.named_steps and hasattr(est.named_steps["frontend"], "feature_names_in_"):
    order = [str(x) for x in est.named_steps["frontend"].feature_names_in_]
    if len(order) == len(feature_cols_str) and set(order) == set(feature_cols_str):
        cur_map = {str(c): c for c in feature_cols}
        feature_cols = [cur_map[x] for x in order]
elif "preprocessor" in est.named_steps and hasattr(est.named_steps["preprocessor"], "feature_names_in_"):
    order = [str(x) for x in est.named_steps["preprocessor"].feature_names_in_]
    if len(order) == len(feature_cols_str) and set(order) == set(feature_cols_str):
        cur_map = {str(c): c for c in feature_cols}
        feature_cols = [cur_map[x] for x in order]

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
fitted = mlca_runtime.fit_record(est, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

if val is not None and len(val):
    X_val = val[feature_cols].copy()
    y_val = val[targets].to_numpy()
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
