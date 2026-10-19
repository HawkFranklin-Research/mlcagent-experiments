import json
from itertools import combinations
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded_cols = protocol["excluded_columns"]
    group_col = protocol["group_column"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol)}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = targets[0]
drop = set(targets) | set(excluded_cols)
if group_col is not None:
    drop.add(group_col)

features = []
excluded_info = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (
        pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        excluded_info.append(f"{c}(key-like)")
        continue
    features.append(c)

tstr = train[target].astype("string").fillna("<NA>")
for c in list(features):
    if train[c].astype("string").fillna("<NA>").equals(tstr):
        features.remove(c)
        excluded_info.append(f"{c}(target copy)")

target_is_num = pd.api.types.is_numeric_dtype(train[target]) and not pd.api.types.is_bool_dtype(train[target])
if target_is_num:
    t = pd.to_numeric(train[target], errors="coerce")
    cand = [
        c for c in features
        if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()
    ]
    found = None
    for k in (1, 2, 3):
        for combo in combinations(cand, k):
            s = sum(train[c] for c in combo)
            if np.allclose(s.astype(float), t.astype(float), equal_nan=True):
                found = combo
                break
        if found:
            break
    if found:
        for c in found:
            if c in features:
                features.remove(c)
                excluded_info.append(f"{c}(target sum)")
    warn = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[[c, target]].corr(numeric_only=True).iloc[0, 1]
            if abs(corr) >= 0.98:
                warn.append(c)
    if warn:
        print("corr>=0.98:", ",".join(warn))

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]
for c in list(cat_cols):
    if train[c].nunique(dropna=True) > max(5, 0.05 * len(train)):
        cat_cols.remove(c)
        features.remove(c)
        excluded_info.append(f"{c}(high-card)")

tokens = ["?", "-", "NA", "N/A", ""]
ph_cols = []
for c in cat_cols:
    stripped = train[c].dropna().astype("string").str.strip()
    if stripped.isin(tokens).any():
        ph_cols.append(c)
for c in cat_cols:
    tr = train[c].astype("string").str.strip()
    va = val[c].astype("string").str.strip()
    train[c] = tr.replace({t: np.nan for t in tokens})
    val[c] = va.replace({t: np.nan for t in tokens})

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop")
expected = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if expected <= 300:
    pre.set_output(transform="pandas")

X_train = train[features].copy()
X_val = val[features].copy()
pre = mlca_runtime.fit_record(pre, X_train, "train")
Xv = pre.transform(X_val)
mlca_runtime.save_model(pre, "preprocessor.pkl")

print("placeholders:", ",".join(ph_cols) if ph_cols else "none", f"({len(ph_cols)} cols)")
print("excluded:", "; ".join(excluded_info) if excluded_info else "none")
print("features:", len(num_cols), "numeric,", len(cat_cols), "categorical")
print("val_transformed:", getattr(Xv, "shape", None))
