import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", []) or []
group_col = protocol.get("group_column")
task = protocol["task"]
y_col = targets[0]
y_train = train[y_col]

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
candidates = [c for c in train.columns if c not in drop_set]

# key-like exclusion + target-equality exclusion
n = len(train)
features, key_drops, deriv_drops = [], [], []
for c in candidates:
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        key_drops.append(c)
        continue
    if s.astype(str).eq(y_train.astype(str)).all():
        deriv_drops.append(c)
        continue
    features.append(c)

num_cols, cat_cols, date_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
        continue
    parsed = pd.to_datetime(s.astype(str), errors="coerce")
    nun = s.nunique(dropna=True)
    if parsed.notna().mean() > 0.9 and nun > 5 and nun < 0.5 * n:
        date_cols.append(c)
        continue
    if nun <= 0.05 * n:
        cat_cols.append(c)

# numeric-target sum check
if pd.api.types.is_numeric_dtype(y_train) and not pd.api.types.is_bool_dtype(y_train):
    nonneg = [c for c in num_cols if (train[c].fillna(0) >= 0).all()]
    for k in (1, 2, 3):
        for combo in combinations(nonneg, k):
            if train[list(combo)].fillna(0).sum(axis=1).eq(y_train).all():
                deriv_drops.extend(combo)
    num_cols = [c for c in num_cols if c not in deriv_drops]

# date derivation
for c in date_cols:
    d = pd.to_datetime(train[c], errors="coerce")
    vd = pd.to_datetime(val[c], errors="coerce")
    for name, fn in [("month", lambda x: x.dt.month),
                     ("dow", lambda x: x.dt.dayofweek),
                     ("hour", lambda x: x.dt.hour)]:
        train[f"{c}_{name}"] = fn(d)
        val[f"{c}_{name}"] = fn(vd)
        num_cols.append(f"{c}_{name}")

# placeholder cleanup on categorical strings
phs = {"", "?", "-", "NA", "N/A"}
ph_cols = []
for c in cat_cols:
    st = train[c].astype(str).str.strip()
    mask = st.isin(phs)
    if mask.any():
        ph_cols.append(c)
    train[c] = st.where(~mask, other=np.nan)
    vs = val[c].astype(str).str.strip()
    vmask = vs.isin(phs)
    val[c] = vs.where(~vmask, other=np.nan)

final_features = num_cols + cat_cols
X_train = train[final_features]
X_val = val[final_features]

cat_width = sum(train[c].nunique(dropna=True) for c in cat_cols) if cat_cols else 0
use_dense = (len(num_cols) + cat_width) <= 300

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_dense)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if use_dense:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, X_train, "train")

# remove any nan-named columns (one-hot can emit if all-missing)
Xt = fitted.transform(X_val)

mlca_runtime.save_model(fitted, "preprocessor.pkl")
print(f"features={len(final_features)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"drop key-like={key_drops} derived={deriv_drops}")
print(f"placeholders in {len(ph_cols)} cols")
print(f"train={X_train.shape} val-enc={Xt.shape}")
mlca_runtime.record_metric("n_features", float(len(final_features)), "train")
