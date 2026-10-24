import mlca_runtime
import json
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

candidate_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group
]

key_like = []
for c in candidate_cols:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.is_unique:
        key_like.append(c)

feature_cols = [c for c in candidate_cols if c not in key_like]

derived = []
for c in feature_cols:
    if train[c].astype(str).equals(train[targets[0]].astype(str)):
        derived.append(c)

feature_cols = [c for c in feature_cols if c not in derived]

num_cols = []
cat_cols = []
date_cols = []
high_card = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) > max(20, int(0.05 * len(train))):
            high_card.append(c)
        else:
            cat_cols.append(c)

if date_cols:
    raise ValueError(f"Datetime feature columns require parsing: {date_cols}")

feature_cols = num_cols + cat_cols

placeholders = {"?", "-", "NA", "N/A"}
placeholder_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders).any() or (stripped == "").any():
            placeholder_cols.append(c)

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])
preprocessor.set_output(transform="pandas")

y_train = train[targets[0]] if len(targets) == 1 else None
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)
X_train_t = preprocessor.transform(X_train)
X_val_t = preprocessor.transform(X_val)

mlca_runtime.record_metric("n_features_raw", len(feature_cols), "train")
mlca_runtime.record_metric("n_features_transformed", X_train_t.shape[1], "train")
mlca_runtime.record_metric("n_train_rows", len(X_train), "train")
mlca_runtime.record_metric("n_val_rows", len(X_val), "val")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"Features: {len(feature_cols)} raw ({len(num_cols)} num, {len(cat_cols)} cat); excluded derivations: {derived}")
print(f"Placeholders: {placeholder_cols if placeholder_cols else 'none'}")
print(f"Transformed train shape: {X_train_t.shape}; val shape: {X_val_t.shape}")
