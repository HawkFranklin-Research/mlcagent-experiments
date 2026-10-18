import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
try:
    targets = set(proto["targets"])
    excluded = set(proto["excluded_columns"])
    group_col = proto["group_column"]
    task = proto["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {sorted(proto.keys())}")

train = pd.read_parquet(inp["train"])
n = len(train)
cand = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

drop_key = []
for c in cand:
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique(dropna=True) == n:
        drop_key.append(c)
    elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == n:
        drop_key.append(c)
features = [c for c in cand if c not in drop_key]

numeric, cat, high = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric.append(c)
    else:
        if s.nunique(dropna=True) <= max(5, 0.05 * n):
            cat.append(c)
        else:
            high.append(c)

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in train.columns:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype("string")
        mask = s.isna() | s.str.strip().isin(ph_tokens) | (s.str.strip() == "")
        if mask.any():
            ph_cols.append(c)

log_cols = []
for c in numeric:
    s = train[c].dropna()
    if len(s) and (s >= 0).all() and s.skew() > 1:
        log_cols.append(c)

transformers = []
num_plain = [c for c in numeric if c not in log_cols]
if num_plain:
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.set_output(transform="pandas")
    transformers.append(("num", pipe, num_plain))
if log_cols:
    pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.set_output(transform="pandas")
    transformers.append(("log", pipe, log_cols))
if cat:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pipe.set_output(transform="pandas")
    transformers.append(("cat", pipe, cat))

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")
X = train[features]
mlca_runtime.fit_record(pre, X, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features={len(features)} numeric={len(numeric)} log={len(log_cols)} cat={len(cat)}")
print(f"dropped_key={len(drop_key)} dropped_high_card={len(high)}")
if ph_cols:
    print(f"placeholder_columns={len(ph_cols)}")
