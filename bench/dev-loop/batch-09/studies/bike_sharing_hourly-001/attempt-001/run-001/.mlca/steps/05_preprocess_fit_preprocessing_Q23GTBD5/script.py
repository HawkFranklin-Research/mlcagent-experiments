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
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"missing {k}; available {list(d)}")
    return d[k]

targets = req(proto, "targets")
excluded = req(proto, "excluded_columns")
task = req(proto, "task")
group = proto.get("group_column")
train = pd.read_parquet(paths["train"])
skip = set(targets) | set(excluded) | ({group} if group else set())

placeholders = {"?", "-", "NA", "N/A", ""}
found = []
for c in train.columns:
    if c in skip or not (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])):
        continue
    s = train[c].astype("string")
    stripped = s.str.strip()
    mask = stripped.isin(placeholders)
    if mask.any():
        found.append(c)
    train[c] = stripped.mask(mask, np.nan).astype("object")
if found:
    print(f"placeholders in {len(found)} cols: {found[:5]}")

cand = [c for c in train.columns if c not in skip]
drop_reason = {}
for c in list(cand):
    s = train[c]
    if s.isna().all():
        drop_reason[c] = "all-missing"
        cand.remove(c)
        continue
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_reason[c] = "datetime"
        cand.remove(c)
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().mean() > 0.9 and s.nunique(dropna=False) > 1:
            drop_reason[c] = "date"
            cand.remove(c)
            continue
        if not s.isna().any() and s.nunique(dropna=False) == len(s):
            drop_reason[c] = "key-like"
            cand.remove(c)
            continue
    elif pd.api.types.is_integer_dtype(s) and not s.isna().any() and s.nunique(dropna=False) == len(s):
        drop_reason[c] = "key-like"
        cand.remove(c)
        continue
for c in list(cand):
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=False) > 0.05 * len(train):
        drop_reason[c] = "high-card"
        cand.remove(c)
if drop_reason:
    print(f"dropped: {list(drop_reason.items())[:5]}")

features = cand[:]
target = targets[0]
y = train[target]
derived = {}
if pd.api.types.is_numeric_dtype(y):
    num = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    for c in num:
        if train[c].equals(y):
            derived[c] = "exact target"
    nonneg = [c for c in num if train[c].min() >= 0]
    for r in (1, 2, 3):
        for combo in combinations(nonneg, r):
            if any(c in derived for c in combo):
                continue
            s = train[list(combo)].sum(axis=1)
            try:
                same = np.allclose(s.to_numpy(dtype=float), y.to_numpy(dtype=float), equal_nan=False)
            except Exception:
                same = False
            if same:
                for c in combo:
                    derived[c] = "target sum"
    features = [c for c in features if c not in derived]
    if derived:
        print(f"excluded derivation: {list(derived.items())[:5]}")

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if pd.notna(corr) and abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print(f"high corr >=0.98 kept: {warn[:5]}")

X = train[features].copy()
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in features if c not in num_cols]
width = len(num_cols) + sum(X[c].nunique(dropna=False) for c in cat_cols)
wide = width > 300
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
    ]), cat_cols))
ct = ColumnTransformer(transformers, remainder="drop")
if not wide:
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass
pre = mlca_runtime.fit_record(ct, X, "train")
if pre is None:
    pre = ct
mlca_runtime.save_model(pre, "preprocessor.pkl")
try:
    enc_width = len(pre.get_feature_names_out())
except Exception:
    enc_width = width
mlca_runtime.record_metric("n_features", float(len(features)), "train")
mlca_runtime.record_metric("encoded_width", float(enc_width), "train")
print(f"fitted preprocessor: {len(features)} features, encoded width {enc_width}")
