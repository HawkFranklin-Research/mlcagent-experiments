import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {sorted(d)}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")

drop_names = set(targets) | set(excluded)
if group is not None:
    drop_names.add(group)
candidates = [c for c in train.columns if c not in drop_names]

feature_cols = []
key_like = []
for c in candidates:
    s = train[c]
    nonnull = s.dropna()
    if len(nonnull) <= 1:
        feature_cols.append(c)
        continue
    is_numeric = pd.api.types.is_numeric_dtype(s)
    is_dt = pd.api.types.is_datetime64_any_dtype(s)
    if not is_numeric and not is_dt and s.nunique(dropna=True) == len(nonnull):
        key_like.append(c)
        continue
    if pd.api.types.is_integer_dtype(s):
        lc = str(c).lower()
        if s.nunique(dropna=True) == len(nonnull) and any(k in lc for k in ("id", "key", "index", "code")):
            key_like.append(c)
            continue
    feature_cols.append(c)
if key_like:
    print("dropped key-like columns: " + ", ".join(map(str, key_like[:10])))

date_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in date_cols:
    dt = pd.to_datetime(train[c], errors="coerce")
    for part, val in [("year", dt.dt.year), ("month", dt.dt.month), ("day", dt.dt.day), ("dow", dt.dt.dayofweek)]:
        name = f"{c}_{part}"
        train[name] = val
        feature_cols.append(name)
feature_cols = [c for c in feature_cols if c not in date_cols]

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

threshold = max(5, int(0.05 * len(train)))
low_cat, high_cat = [], []
for c in cat_cols:
    n = train[c].nunique(dropna=True)
    (low_cat if n <= threshold and n <= 50 else high_cat).append(c)
if high_cat:
    print(f"dropped high-cardinality text/code columns: {len(high_cat)}")
    feature_cols = [c for c in feature_cols if c not in high_cat]
    cat_cols = low_cat

placeholder_tokens = {"?", "-", "NA", "N/A"}
found_tokens = set()
for c in cat_cols:
    vals = train[c].dropna().astype(str).str.strip()
    for tok in placeholder_tokens:
        if (vals == tok).any():
            found_tokens.add(tok)
    if (vals == "").any():
        found_tokens.add("whitespace")
if found_tokens:
    print("placeholder tokens found: " + ", ".join(sorted(found_tokens)))
    for c in cat_cols:
        train[c] = train[c].astype("string").str.strip().replace(list(placeholder_tokens) + [""], pd.NA)

log_cols, plain_num = [], []
for c in num_cols:
    s = train[c].dropna()
    sk = s.skew() if len(s) > 1 else np.nan
    if len(s) > 1 and (s >= 0).all() and not pd.isna(sk) and sk > 2:
        log_cols.append(c)
    else:
        plain_num.append(c)

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in low_cat)
cat_sparse = width > 300
transformers = []
if log_cols:
    transformers.append(("num_log", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ]), log_cols))
if plain_num:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), plain_num))
if low_cat:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=cat_sparse)),
    ]), low_cat))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if width <= 300:
    pre.set_output(transform="pandas")

X_train = train[feature_cols].copy()
pre = mlca_runtime.fit_record(pre, X_train, partition="train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")
print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(low_cat)} train_rows={len(train)}")
