import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in prot:
        raise KeyError(f"missing {k}; available: {list(prot)}")
targets = prot["targets"]
excluded = set(prot["excluded_columns"])
group = prot.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

placeholder_cols = 0
ph_tokens = ["?", "-", "NA", "N/A", ""]
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        try:
            s = train[c].astype("object")
            stripped = s.str.strip()
            found = stripped.isin(ph_tokens)
            if found.any():
                placeholder_cols += 1
                train.loc[found, c] = np.nan
            train[c] = stripped.mask(found, np.nan)
            if c in val.columns:
                sv = val[c].astype("object")
                stripped_v = sv.str.strip()
                found_v = stripped_v.isin(ph_tokens)
                if found_v.any():
                    val.loc[found_v, c] = np.nan
                val[c] = stripped_v.mask(found_v, np.nan)
        except Exception:
            pass
print(f"placeholder columns: {placeholder_cols}")

date_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        parsed = pd.to_datetime(train[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            date_cols.append(c)
for c in date_cols:
    for df in (train, val):
        if c not in df.columns:
            continue
        dt = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"] = dt.dt.year
        df[f"{c}_month"] = dt.dt.month
        df[f"{c}_day"] = dt.dt.day
        df[f"{c}_dow"] = dt.dt.dayofweek
print(f"date columns parsed: {len(date_cols)}")

base = [c for c in train.columns if c not in targets and c not in excluded and c != group and c not in date_cols]
feat = []
for c in base:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique(dropna=True) == len(train):
            continue
    if pd.api.types.is_integer_dtype(train[c]) and train[c].nunique(dropna=True) == len(train):
        continue
    feat.append(c)
feat = [c for c in feat if train[c].notna().any()]

y = train[targets[0]]
drop = set()
for c in feat:
    if pd.api.types.is_numeric_dtype(train[c]):
        if np.allclose(train[c].astype(float), y.astype(float), equal_nan=False):
            drop.add(c)
num_nonneg = [c for c in feat if c not in drop and pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
if len(num_nonneg) <= 20:
    for r in (2, 3):
        for combo in combinations(num_nonneg, r):
            mask = train[list(combo)].notna().all(axis=1) & y.notna()
            if mask.all() and np.allclose(train[list(combo)].sum(axis=1).astype(float), y.astype(float)):
                drop.update(combo)
for c in sorted(drop):
    print(f"excluded derivation: {c}")
feat = [c for c in feat if c not in drop]

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in feat:
        if pd.api.types.is_numeric_dtype(train[c]):
            v = train[c].corr(y)
            if pd.notna(v) and abs(v) >= 0.98:
                warn.append(c)
if warn:
    print("high corr: " + ", ".join(warn))

num_cols, cat_cols = [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        u = s.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(s) and u <= 12 and not set(s.dropna().unique()).issubset({0, 1}):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        u = s.nunique(dropna=True)
        if u <= max(20, int(0.05 * len(train))):
            cat_cols.append(c)
print(f"features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

X_train = train[num_cols + cat_cols].copy()
width = len(num_cols) + sum(train[c].nunique(dropna=True) + 1 for c in cat_cols)
sparse = width > 300
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    ct.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
pre = mlca_runtime.fit_record(Pipeline([("ct", ct)]) if sparse else ct, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
