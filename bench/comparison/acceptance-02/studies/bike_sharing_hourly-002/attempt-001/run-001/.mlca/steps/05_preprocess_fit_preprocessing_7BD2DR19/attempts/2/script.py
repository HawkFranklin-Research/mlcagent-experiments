import json
from itertools import combinations

import numpy as np
import pandas as pd

import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def req(key):
    if key not in protocol:
        raise KeyError(f"missing {key}; available={list(protocol)}")
    return protocol[key]


targets = req("targets")
excluded = req("excluded_columns")
group = req("group_column")
task = req("task")
train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
base_cols = [c for c in train.columns if c not in drop]


def is_keylike(s):
    if s.isna().any():
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False


feature_cols = [c for c in base_cols if not is_keylike(train[c])]
X = train.copy()

time_cols = set(task.get("time_columns", []))
for c in list(feature_cols):
    if c in time_cols and not pd.api.types.is_numeric_dtype(X[c]):
        dt = pd.to_datetime(X[c], errors="coerce")
        if dt.notna().mean() >= 0.9:
            for suf, ser in (
                ("year", dt.dt.year),
                ("month", dt.dt.month),
                ("day", dt.dt.day),
                ("dow", dt.dt.dayofweek),
            ):
                X[f"{c}_{suf}"] = ser
            feature_cols.remove(c)
            feature_cols.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dow"])

y = train[targets[0]]
num_candidates = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
derived = set()
for r in (1, 2, 3):
    for combo in combinations(num_candidates, r):
        if any(X[c].min() < 0 for c in combo):
            continue
        if (X[list(combo)].sum(axis=1) == y).all():
            derived.update(combo)

corr_warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in num_candidates:
        if c in derived:
            continue
        if X[c].nunique(dropna=True) > 1:
            corr = X[c].corr(y)
            if pd.notna(corr) and abs(corr) >= 0.98:
                corr_warn.append(c)

feature_cols = [c for c in feature_cols if c not in derived]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

high = [c for c in cat_cols if X[c].nunique(dropna=True) > 0.05 * len(X)]
cat_cols = [c for c in cat_cols if c not in high]

placeholder_cols = []
for c in cat_cols:
    s = X[c].astype("string")
    stripped = s.str.strip()
    bad = stripped.isin(["?", "-", "NA", "N/A"]) | (stripped.fillna("") == "")
    if bad.any():
        placeholder_cols.append(c)
    X[c] = stripped.mask(bad, np.nan)

feature_cols = num_cols + cat_cols
X_train = X[feature_cols].copy()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer(
    [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
width = len(num_cols) + sum(X_train[c].nunique(dropna=True) for c in cat_cols)
if width <= 300:
    preprocessor.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

lines = [f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}"]
if derived:
    lines.append("derived excluded: " + ", ".join(sorted(derived)))
if corr_warn:
    lines.append("corr>=0.98: " + ", ".join(corr_warn))
if placeholder_cols:
    lines.append("placeholders: " + ", ".join(placeholder_cols))
if high:
    lines.append("high-card dropped: " + ", ".join(high))
for line in lines[:5]:
    print(line)
