import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
candidate_cols = [c for c in train.columns if c not in drop_cols]

feature_cols = []
for c in candidate_cols:
    s = train[c]
    if s.dtype.kind in "iu" and s.nunique() == len(train) and s.nunique() > 1:
        continue
    if s.dtype.kind in "O" or str(s.dtype) == "string":
        if s.nunique() == len(train):
            continue
    feature_cols.append(c)

excluded_features = []
for c in feature_cols:
    for t in targets:
        if train[c].astype(str).equals(train[t].astype(str)):
            excluded_features.append((c, f"exact equality with target {t}"))
            break
for c, reason in excluded_features:
    print(f"Excluded {c}: {reason}")
feature_cols = [c for c in feature_cols if c not in [e[0] for e in excluded_features]]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]

for c in list(cat_cols):
    if train[c].nunique() > 0.05 * len(train):
        print(f"Dropping high-cardinality {c}: {train[c].nunique()} unique")
        cat_cols.remove(c)
        feature_cols.remove(c)

placeholder_counts = {}
for c in cat_cols:
    s = train[c].astype(str)
    for ph in ["?", "-", "NA", "N/A"]:
        if (s == ph).any():
            placeholder_counts.setdefault(ph, []).append(c)
    if s.str.strip().eq("").any():
        placeholder_counts.setdefault("whitespace", []).append(c)
if placeholder_counts:
    print(f"Placeholders found: {placeholder_counts}")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

X_train = train[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, "train")

X_val = val[feature_cols]
X_val_trans = preprocessor.transform(X_val)
print(f"Features: {len(feature_cols)} (numeric={len(numeric_cols)}, categorical={len(cat_cols)})")
print(f"Transformed val shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
