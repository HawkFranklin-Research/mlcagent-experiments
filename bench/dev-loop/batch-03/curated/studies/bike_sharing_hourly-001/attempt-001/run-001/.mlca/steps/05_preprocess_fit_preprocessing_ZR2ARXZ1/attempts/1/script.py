import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    n = len(s)
    if s.dtype == object or str(s.dtype).startswith("str"):
        if s.nunique(dropna=True) == n:
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == n:
            continue
    feature_cols.append(c)

for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("str"):
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.9:
            feature_cols.remove(c)

y = train[targets[0]]

excluded_feats = []
for c in list(feature_cols):
    if train[c].equals(y):
        excluded_feats.append((c, "exact equality with target"))
        feature_cols.remove(c)

numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for i, c1 in enumerate(numeric_feats):
    if c1 not in feature_cols:
        continue
    for c2 in numeric_feats[i+1:]:
        if c2 not in feature_cols:
            continue
        if (train[c1] + train[c2]).equals(y):
            excluded_feats.append((c1, f"sum with {c2} equals target"))
            feature_cols.remove(c1)
            break

corr_warnings = []
if task.get("type") == "regression":
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = train[c].corr(y)
            if abs(r) >= 0.98:
                corr_warnings.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("str"):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders).any():
            placeholder_cols.append(c)

n_rows = len(train)
threshold = 0.05 * n_rows
numeric_cols = []
cat_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > threshold:
            continue
        cat_cols.append(c)

n_out = len(numeric_cols)
for c in cat_cols:
    n_out += train[c].nunique(dropna=True)

if n_out <= 300:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")

if n_out <= 300:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
preprocessor.fit(X_train)

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

if excluded_feats:
    print(f"Excluded features: {excluded_feats}")
if corr_warnings:
    print(f"Correlation warnings (|r|>=0.98): {corr_warnings}")
if placeholder_cols:
    print(f"Placeholder tokens in {len(placeholder_cols)} columns: {placeholder_cols}")
print(f"Fitted preprocessor on {X_train.shape[0]} rows, {len(feature_cols)} features")
