import json
from itertools import combinations

import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
protocol = json.loads(paths["_protocol"].read_text())
try:
    targets = list(protocol["targets"])
    excluded = set(protocol["excluded_columns"])
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available keys: {list(protocol)}") from exc
group = protocol.get("group_column")
target_set = set(targets) | excluded

def key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and not s.isna().any()
    return False

features = [
    c for c in train.columns
    if c not in target_set and c != group and not key_like(train[c])
]
if not features:
    raise ValueError("No feature columns remain after protocol/key exclusions")

derived = set()
for t in targets:
    y = train[t]
    if not pd.api.types.is_numeric_dtype(y):
        continue
    for c in features:
        if c not in derived and pd.api.types.is_numeric_dtype(train[c]):
            if (train[c] == y).all():
                derived.add(c)
                print(f"excluded derived feature {c} == {t}")
    num_feats = [
        c for c in features
        if c not in derived and pd.api.types.is_numeric_dtype(train[c])
        and (train[c].dropna() >= 0).all()
    ]
    if len(num_feats) <= 30:
        found = False
        for r in (2, 3):
            if found:
                break
            for combo in combinations(num_feats, r):
                if (train[list(combo)].sum(axis=1) == y).all():
                    derived.update(combo)
                    print(f"excluded derived sum {combo} == {t}")
                    found = True
                    break
features = [c for c in features if c not in derived]

corr_warn = []
for t in targets:
    if not pd.api.types.is_numeric_dtype(train[t]):
        continue
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = train[c].corr(train[t])
            if pd.notna(r) and abs(r) >= 0.98:
                corr_warn.append(c)
if corr_warn:
    print(f"correlation warning: {sorted(set(corr_warn))}")

ph = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
ph_cols = []
for c in text_cols:
    s = train[c].astype("string")
    stripped = s.str.strip()
    if stripped.isin(ph | {" "}).any():
        ph_cols.append(c)
        train[c] = stripped.mask(stripped.isin(ph) | stripped.str.fullmatch(r"\s*"), pd.NA)
if ph_cols:
    print(f"placeholders found in {len(ph_cols)} text columns")

n_rows = len(train)
num_cols, cat_cols, high_card = [], [], []
for c in features:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s):
        if s.nunique(dropna=True) > max(5, int(0.05 * n_rows)):
            high_card.append(c)
        else:
            cat_cols.append(c)
    elif pd.api.types.is_integer_dtype(s):
        vals = set(s.dropna().unique())
        if len(vals) <= 10 and not vals.issubset({0, 1}):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        num_cols.append(c)
features = [c for c in features if c not in high_card]
if high_card:
    print(f"dropped high-cardinality {sorted(high_card)}")

cat_width = sum(max(1, train[c].nunique(dropna=True)) for c in cat_cols)
wide = len(num_cols) + cat_width > 300
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])
transformers = []
if num_cols:
    transformers.append(("num", numeric_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not wide:
    pre.set_output(transform="pandas")
X_train = train[features].copy()
pre = mlca_runtime.fit_record(pre, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"fitted features={len(features)} numeric={len(num_cols)} categorical={len(cat_cols)}")
