import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
base_exclude = set(targets + excluded + ([group] if group else []))

def keylike(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.notna().all() and s.nunique(dropna=True) == len(s) and ("id" in str(s.name).lower() or s.is_monotonic_increasing)
    return False

features = [c for c in train.columns if c not in base_exclude and not keylike(train[c])]

# Exact target-derivation screen.
target = train[targets[0]]
excluded_features = []
tgt_s = target.astype("string").fillna("<NA>")
for c in list(features):
    if train[c].astype("string").fillna("<NA>").equals(tgt_s):
        excluded_features.append(c)
        features.remove(c)

# Numeric row-sum derivation screen.
if pd.api.types.is_numeric_dtype(target):
    num_feats = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for r in (1, 2, 3):
        for combo in itertools.combinations(num_feats, r):
            if (train[list(combo)].sum(axis=1) == target).all():
                for c in combo:
                    if c in features:
                        features.remove(c)
                        excluded_features.append(c)

# Exact-equality only for categorical targets; numeric correlation warnings otherwise.
if pd.api.types.is_numeric_dtype(target):
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            if abs(train[c].corr(target)) >= 0.98:
                print(f"correlation warning: {c}")

print(f"excluded features: {excluded_features}")

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in features:
    s = train[c].astype("string")
    found = s.str.strip().isin(placeholders) | (s.str.strip().str.len() == 0)
    if found.any():
        ph_cols.append(c)

def clean_text(df):
    out = df.copy()
    for c in features:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            s = out[c].astype("string").str.strip()
            s = s.mask(s.isin(placeholders) | (s.str.len() == 0), pd.NA)
            out[c] = s.astype(object).where(s.notna(), np.nan)
    return out

train_c = clean_text(train)
val_c = clean_text(val)
print(f"placeholder columns: {len(ph_cols)}")

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train_c[c])]
cat_cols = [c for c in features if c not in numeric_cols]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
width = len(numeric_cols) + sum(train_c[c].nunique(dropna=True) for c in cat_cols)
pre = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")
if width <= 300:
    pre.set_output(transform="pandas")

pre = mlca_runtime.fit_record(pre, train_c[features], "train")
Xv = pre.transform(val_c[features])
print(f"train features: {len(features)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"val transformed shape: {Xv.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
