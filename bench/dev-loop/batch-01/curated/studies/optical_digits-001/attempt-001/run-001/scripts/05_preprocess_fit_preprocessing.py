import json
import mlca_runtime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "excluded_columns", "group_column"):
    if k not in protocol:
        raise KeyError(f"Missing '{k}' in _protocol. Available keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = [c for c in train.columns if c not in drop]

# key-like screening
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object and s.nunique(dropna=True) == len(s):
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        u = s.dropna().unique()
        if len(u) == len(s) and set(u) == set(range(len(s))):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# placeholder scan for text columns
ph_cols = 0
for c in feature_cols:
    s = train[c]
    if s.dtype == object:
        vals = s.dropna().astype(str)
        if vals.str.strip().isin(["?", "-", "NA", "N/A", ""]).any():
            ph_cols += 1
print(f"placeholder columns: {ph_cols}")

# type sorting
n_rows = len(train)
numeric_cols, cat_cols, drop_high = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(1, int(0.05 * n_rows)):
            cat_cols.append(c)
        else:
            drop_high.append(c)

# RL-0008: no targets -> no derivations to exclude
print(f"features: {len(feature_cols)} numeric={len(numeric_cols)} cat={len(cat_cols)} dropped_high_card={len(drop_high)}")

transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), numeric_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")

X_train = train[feature_cols]
mlca_runtime.fit_record(pre, X_train, partition="train")

mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"fitted preprocessor on {X_train.shape[0]} rows, {X_train.shape[1]} features")
