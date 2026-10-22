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
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
drop_cols = set(targets) | set(excluded)
if group is not None:
    drop_cols.add(group)

feature_cols = [c for c in train.columns if c not in drop_cols]
n = len(train)
key_like = []
for c in list(feature_cols):
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == n:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == n:
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        mask = stripped.isin(placeholders) | (stripped == "")
        if mask.any():
            ph_cols += 1
            train[c] = stripped.replace(list(placeholders), np.nan)

target = targets[0]
y = train[target]
numeric_candidates = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
derived = set()
for c in numeric_candidates:
    if np.allclose(train[c].astype(float), y.astype(float), rtol=1e-9, atol=1e-9):
        derived.add(c)
nonneg = [c for c in numeric_candidates if (train[c] >= 0).all()]
for k in (1, 2, 3):
    for combo in combinations(nonneg, k):
        s = train[list(combo)].sum(axis=1)
        if np.allclose(s.astype(float), y.astype(float), rtol=1e-9, atol=1e-9):
            derived.update(combo)
if derived:
    print("Derived features excluded: " + ", ".join(sorted(derived)))
feature_cols = [c for c in feature_cols if c not in derived]

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]:
        corr = train[[c, target]].dropna().corr().iloc[0, 1]
        if abs(corr) >= 0.98:
            warn.append(c)
if warn:
    print("High correlation >=0.98 (kept): " + ", ".join(sorted(warn)))

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part, vals in [("year", s.dt.year), ("month", s.dt.month), ("dayofweek", s.dt.dayofweek), ("hour", s.dt.hour)]:
            new = f"{c}_{part}"
            train[new] = vals
            numeric_cols.append(new)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        if s.nunique(dropna=True) > 0.05 * n:
            print(f"Dropped high-cardinality {c} ({s.nunique(dropna=True)} unique)")
        else:
            cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)

X_train = train[numeric_cols + cat_cols].copy()
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if est_width <= 300:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, X_train, "train", y=y)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Placeholder columns: {ph_cols}; numeric: {len(numeric_cols)}; categorical: {len(cat_cols)}")
