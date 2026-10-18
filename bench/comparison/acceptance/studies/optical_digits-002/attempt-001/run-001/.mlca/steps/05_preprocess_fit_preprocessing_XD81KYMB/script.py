import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from scipy.stats import skew

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"Missing '{k}' in protocol; available keys: {list(protocol.keys())}")

targets = list(protocol.get("targets") or [])
excluded = set(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
protected = set(targets) | excluded | ({group} if group else set())

feature_cols = []
for c in train.columns:
    if c in protected:
        continue
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_integer_dtype(s) and s.notna().all():
        if s.nunique() == len(s):
            continue
    feature_cols.append(c)

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

log_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 10 and s.min() >= 0:
        try:
            sk = float(skew(s))
        except Exception:
            continue
        if np.isfinite(sk) and sk > 2:
            log_cols.append(c)

num_log_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p)),
    ("scale", StandardScaler()),
])
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if log_cols:
    transformers.append(("num_log", num_log_pipe, log_cols))
plain_num = [c for c in numeric_cols if c not in log_cols]
if plain_num:
    transformers.append(("num", num_pipe, plain_num))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

ct = ColumnTransformer(transformers=transformers, remainder="drop")
feature_df = train[feature_cols]
mlca_runtime.fit_record(ct, feature_df, partition="train")
probe = ct.transform(feature_df.head(2))
width = probe.shape[1] if hasattr(probe, "shape") else len(probe)
if width <= 300:
    for p in [num_log_pipe, num_pipe, cat_pipe]:
        p.set_output(transform="pandas")
    ct.set_output(transform="pandas")

mlca_runtime.save_model(ct, "preprocessor.pkl")
mlca_runtime.record_metric("n_features_in", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_features_out", float(width), "train")
print(f"features_in={len(feature_cols)} features_out={width} numeric={len(numeric_cols)} categorical={len(cat_cols)} log1p={len(log_cols)}")
print(f"protected={sorted(protected)}")
