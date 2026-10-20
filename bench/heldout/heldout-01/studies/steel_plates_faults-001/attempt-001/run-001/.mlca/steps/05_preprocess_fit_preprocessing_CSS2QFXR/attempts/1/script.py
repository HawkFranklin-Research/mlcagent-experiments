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
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())

if "targets" not in protocol:
    raise ValueError(f"Missing targets; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)
drop = set(targets) | set(excluded) | ({group} if group else set())
candidate = [c for c in train.columns if c not in drop]

features = []
for c in candidate:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(train):
        continue
    features.append(c)

placeholders = {"?", "-", "NA", "N/A", "", " "}
ph_cols = 0
for c in features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholders) | stripped.fillna("").str.fullmatch(r"\s*")
        if mask.any():
            ph_cols += 1
            train[c] = stripped.mask(mask, pd.NA)
        else:
            train[c] = stripped
if ph_cols:
    print(f"placeholders found in {ph_cols} columns")

new_feats = []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        try:
            dt = pd.to_datetime(s, errors="coerce")
        except Exception:
            dt = pd.Series([pd.NaT] * len(train), index=train.index)
        if not (dt.notna().mean() > 0.9 and dt.nunique() > 5):
            new_feats.append(c)
            continue
    else:
        new_feats.append(c)
        continue
    for part, func in (("hour", lambda x: x.dt.hour), ("dayofweek", lambda x: x.dt.dayofweek), ("month", lambda x: x.dt.month)):
        train[f"{c}_{part}"] = func(dt)
        new_feats.append(f"{c}_{part}")
features = new_feats

exclusions = []
to_drop = set()
for t in targets:
    if t not in train:
        continue
    y = train[t]
    for f in list(features):
        if f not in train:
            continue
        if train[f].equals(y):
            to_drop.add(f)
            exclusions.append(f"{f}=exact({t})")
    nums = [f for f in features if f not in to_drop and pd.api.types.is_numeric_dtype(train[f]) and (train[f].dropna() >= 0).all()]
    for k in (2, 3):
        for combo in combinations(nums, k):
            if not all(c in features for c in combo):
                continue
            s = train[list(combo)].sum(axis=1)
            if (s == y).all():
                to_drop.update(combo)
                exclusions.append(f"{'+'.join(combo)}=sum({t})")
features = [f for f in features if f not in to_drop]
if exclusions:
    print("excluded derivations: " + ", ".join(exclusions))

warn = []
for t in targets:
    if t not in train or not pd.api.types.is_numeric_dtype(train[t]):
        continue
    y = train[t]
    if y.std() == 0:
        continue
    for f in features:
        if f not in train or not pd.api.types.is_numeric_dtype(train[f]) or train[f].std() == 0:
            continue
        corr = train[f].corr(y)
        if abs(corr) >= 0.98:
            warn.append(f)
if warn:
    print("high corr kept: " + ", ".join(sorted(set(warn))))

num_cols, cat_cols = [], []
for f in features:
    s = train[f]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(f)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) > max(5, 0.05 * len(train)):
            continue
        cat_cols.append(f)
    else:
        try:
            train[f] = pd.to_numeric(train[f], errors="coerce")
            num_cols.append(f)
        except Exception:
            continue

width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
use_pandas = width <= 300
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop")
if use_pandas:
    ct.set_output(transform="pandas")

X = train[num_cols + cat_cols].copy()
mlca_runtime.fit_record(ct, X, "train")
mlca_runtime.save_model(ct, "preprocessor.pkl")
print(f"train shape {train.shape}; features {len(num_cols) + len(cat_cols)}; encoded width ~{width}")
mlca_runtime.record_metric("n_features", len(num_cols) + len(cat_cols), "train")
mlca_runtime.record_metric("encoded_width", width, "train")
