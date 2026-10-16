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
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop_cols = set(targets) | set(excluded)
if group:
    drop_cols.add(group)
feature_cols = [c for c in train.columns if c not in drop_cols]

target_col = targets[0]
for c in list(feature_cols):
    if train[c].equals(train[target_col]):
        print(f"Excluded {c}: exact match with target")
        feature_cols.remove(c)

num_cols = []
cat_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        num_cols.append(c)
    else:
        if train[c].nunique() > 0.05 * len(train):
            continue
        cat_cols.append(c)

placeholders = ['?', '-', 'NA', 'N/A']
found_cols = []
for c in cat_cols:
    s = train[c].astype(str).str.strip()
    if s.isin(placeholders).any() or (s == '').any():
        found_cols.append(c)
if found_cols:
    print(f"Placeholders found in {len(found_cols)} columns: {found_cols[:5]}")
for c in cat_cols:
    train[c] = train[c].astype(str).str.strip().replace(placeholders + [''], np.nan)
    val[c] = val[c].astype(str).str.strip().replace(placeholders + [''], np.nan)

print(f"Features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')
preprocessor.set_output(transform='pandas')

mlca_runtime.fit_record(preprocessor, train[feature_cols], "train", train[target_col])
transformed = preprocessor.transform(val[feature_cols])
print(f"Transformed val shape: {transformed.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
