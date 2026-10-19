import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])

# Feature contract per RL-0007
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidates = [c for c in train.columns if c not in drop]
feature_cols = []
n = len(train)
for c in candidates:
    s = train[c]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        if s.nunique() == n:
            continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique() == n and s.min() >= 0 and s.max() <= n * 2:
            continue
    feature_cols.append(c)

if not feature_cols:
    raise ValueError("No feature columns determined.")

# Categorize features
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

transformers = []
if numeric_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")

# Set output format if width <= 300
width = len(numeric_cols) + sum(train[c].nunique() for c in cat_cols)
if width <= 300:
    preprocessor.set_output(transform="pandas")

pipeline = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])
if width <= 300:
    pipeline.set_output(transform="pandas")

X_train = train[feature_cols]
y_train = train[targets]

mlca_runtime.fit_record(pipeline, X_train, partition="train", y=y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"Fitted RandomForest on {len(X_train)} rows, {len(feature_cols)} features.")
print("Saved alternative_model.pkl")
