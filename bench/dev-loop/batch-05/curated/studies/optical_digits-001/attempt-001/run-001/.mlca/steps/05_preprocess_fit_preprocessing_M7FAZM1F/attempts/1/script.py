import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = list(protocol.get("targets", []))
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if not s.isna().any() and s.nunique() == len(s) and not pd.api.types.is_float_dtype(s):
        print(f"excluded key-like: {c}")
        continue
    feat_cols.append(c)

ph_tokens = ['?', '-', 'NA', 'N/A', '']
ph_cols = 0
for c in feat_cols:
    if train[c].dtype == object:
        s = train[c].astype(str).str.strip()
        if s.isin(ph_tokens).any():
            ph_cols += 1
        train[c] = s.replace(ph_tokens, np.nan)
        val[c] = val[c].astype(str).str.strip().replace(ph_tokens, np.nan)
if ph_cols:
    print(f"placeholder tokens found in {ph_cols} columns")

numeric_cols, cat_cols, drop_hi = [], [], []
for c in feat_cols:
    s = train[c]
    if s.dtype == object:
        nu = s.nunique()
        if nu / max(len(s), 1) > 0.05 and nu > 10:
            drop_hi.append(c)
        else:
            cat_cols.append(c)
    else:
        numeric_cols.append(c)
for c in drop_hi:
    print(f"excluded high-cardinality: {c}")

log_cols, plain_num = [], []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) and s.min() >= 0 and s.skew() > 2:
        log_cols.append(c)
    else:
        plain_num.append(c)

log_pipe = Pipeline([
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
])
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if plain_num:
    transformers.append(("num", num_pipe, plain_num))
if log_cols:
    transformers.append(("lognum", log_pipe, log_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

ct = ColumnTransformer(transformers=transformers, remainder="drop")
ct.set_output(transform="pandas")

X_train = train[plain_num + log_cols + cat_cols].copy()
mlca_runtime.fit_record(ct, X_train, "train")
mlca_runtime.save_model(ct, "preprocessor.pkl")

n_out = len(ct.get_feature_names_out())
print(f"features used: {len(plain_num)} numeric + {len(log_cols)} log1p + {len(cat_cols)} categorical -> {n_out} encoded")
