import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    task = protocol["task"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {sorted(protocol)}")
group = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
candidate = [c for c in train.columns if c not in drop]

key_like = []
for c in candidate:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=False) == len(s):
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        key_like.append(c)
candidate = [c for c in candidate if c not in key_like]

derived = set()
for t in targets:
    if t not in train:
        continue
    y = train[t]
    nonneg = [c for c in candidate if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for r in range(1, 4):
        for cols in combinations(nonneg, r):
            if t not in cols and (train[list(cols)].sum(axis=1) == y).all():
                derived.update(cols)
for c in candidate:
    if any(c != t and train[c].equals(train[t]) for t in targets if t in train):
        derived.add(c)
candidate = [c for c in candidate if c not in derived]

placeholders = []
for c in candidate:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(["", "?", "-", "NA", "N/A"]).any():
            placeholders.append(c)

date_cols = [c for c in candidate if pd.api.types.is_datetime64_any_dtype(train[c])]
numeric_cols = [c for c in candidate if c not in date_cols and pd.api.types.is_numeric_dtype(train[c])]
cat_cols = []
high_card = []
for c in candidate:
    if c in numeric_cols or c in date_cols:
        continue
    nun = train[c].nunique(dropna=False)
    if nun > max(1, int(0.05 * len(train))):
        high_card.append(c)
    else:
        cat_cols.append(c)
feature_cols = numeric_cols + cat_cols

log_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) and s.min() >= 0 and s.skew() > 1.0:
        log_cols.append(c)
plain_num = [c for c in numeric_cols if c not in log_cols]

n_out = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
dense = n_out <= 300
trans = []
if log_cols:
    trans.append(("num_log", Pipeline([("imputer", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scaler", StandardScaler())]), log_cols))
if plain_num:
    trans.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), plain_num))
if cat_cols:
    trans.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), cat_cols))

pre = ColumnTransformer(trans, remainder="drop", verbose_feature_names_out=False)
if dense:
    pre.set_output(transform="pandas")
mlca_runtime.fit_record(pre, train[feature_cols], partition="train")
mlca_runtime.save_model(pre, "preprocessor.pkl")

print(f"features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
if key_like:
    print("key-like dropped: " + ", ".join(key_like))
if high_card:
    print("high-card dropped: " + ", ".join(high_card))
if log_cols:
    print("log1p: " + ", ".join(log_cols))
if placeholders:
    print(f"placeholders in {len(placeholders)} columns")
