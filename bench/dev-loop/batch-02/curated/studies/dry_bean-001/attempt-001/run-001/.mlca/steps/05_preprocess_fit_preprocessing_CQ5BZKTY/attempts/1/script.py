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
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}'. Available keys: {list(d.keys())}")
    return d[key]

targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group_col = get_key(protocol, "group_column")
task = get_key(protocol, "task")

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]

all_cols = list(train.columns)
drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)
feature_cols = [c for c in all_cols if c not in drop_cols]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith('string'):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s) and (c.lower().endswith('id') or c.lower().endswith('code') or c.lower().endswith('key')):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]
if key_like:
    print(f"Dropped key-like columns: {key_like}")

excluded_features = []
for c in feature_cols:
    if train[c].astype(str).equals(train[targets[0]].astype(str)):
        excluded_features.append((c, "exact equality with target"))
if pd.api.types.is_numeric_dtype(train[targets[0]]):
    target_vals = train[targets[0]]
    nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(nonneg, r):
            if (train[list(combo)].sum(axis=1) == target_vals).all():
                excluded_features.append((combo, "row sum equals target"))
                break
        if excluded_features:
            break
if excluded_features:
    for c, reason in excluded_features:
        print(f"Excluded feature {c}: {reason}")
    excluded_names = set()
    for c, _ in excluded_features:
        if isinstance(c, tuple):
            excluded_names.update(c)
        else:
            excluded_names.add(c)
    feature_cols = [c for c in feature_cols if c not in excluded_names]
else:
    print("No target-derived features excluded.")

placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = []
for c in feature_cols:
    if train[c].dtype == object or str(train[c].dtype).startswith('string'):
        vals = train[c].astype(str).str.strip()
        if vals.isin(placeholders).any():
            placeholder_cols.append(c)
            for df in [train, val]:
                df[c] = df[c].astype(str).str.strip().replace(list(placeholders), np.nan)
if placeholder_cols:
    print(f"Placeholder tokens found in {len(placeholder_cols)} columns: {placeholder_cols}")
else:
    print("No placeholder tokens found in text columns.")

numeric_features = []
categorical_features = []
high_card_dropped = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_features.append(c)
    else:
        n_unique = s.nunique(dropna=False)
        if n_unique <= max(5, 0.05 * len(train)):
            categorical_features.append(c)
        else:
            high_card_dropped.append(c)
if high_card_dropped:
    print(f"Dropped high-cardinality text columns: {high_card_dropped}")

final_features = numeric_features + categorical_features

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numeric_transformer.set_output(transform='pandas')
categorical_transformer.set_output(transform='pandas')

transformers = []
if numeric_features:
    transformers.append(('num', numeric_transformer, numeric_features))
if categorical_features:
    transformers.append(('cat', categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
preprocessor.set_output(transform='pandas')

X_train = train[final_features]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)

X_val = val[final_features]
val_transformed = preprocessor.transform(X_val)
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Transformed val shape: {val_transformed.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
