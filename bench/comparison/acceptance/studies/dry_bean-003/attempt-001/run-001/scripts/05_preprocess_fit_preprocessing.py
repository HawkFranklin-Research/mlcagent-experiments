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
protocol = json.loads(inputs['_protocol'].read_text())

def get_key(proto, key):
    if key not in proto:
        raise KeyError(f"Missing key '{key}' in _protocol. Available top-level keys: {list(proto.keys())}")
    return proto[key]

targets = get_key(protocol, 'targets')
excluded = get_key(protocol, 'excluded_columns')
group_col = get_key(protocol, 'group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets + excluded + [group_col]]
for c in list(feature_cols):
    if train[c].dtype == 'object' and train[c].nunique() == len(train):
        feature_cols.remove(c)
        print(f"Excluded key-like column: {c}")
        continue

target_col = targets[0]
if target_col in train.columns:
    for c in list(feature_cols):
        if train[c].equals(train[target_col]):
            feature_cols.remove(c)
            print(f"Excluded derived feature: {c} (exact match with target)")

numeric_features = []
categorical_features = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_features.append(c)
    else:
        nunique = train[c].nunique()
        if nunique > 0.05 * len(train):
            print(f"Dropped high-cardinality text/code column: {c}")
        else:
            categorical_features.append(c)

placeholder_tokens = {'?', '-', 'NA', 'N/A'}
placeholder_cols = 0
for c in categorical_features:
    if train[c].dtype == 'object':
        s = train[c].astype(str).str.strip()
        if s.isin(placeholder_tokens).any() or s.eq('').any():
            placeholder_cols += 1
print(f"Placeholder tokens found in {placeholder_cols} text columns.")

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
width = len(numeric_features) + sum(train[c].nunique() for c in categorical_features)
if width <= 300:
    preprocessor.set_output(transform='pandas')
    for name, trans, cols in preprocessor.transformers:
        if hasattr(trans, 'set_output'):
            trans.set_output(transform='pandas')

X_train = train[numeric_features + categorical_features].copy()
y_train = train[target_col] if target_col in train.columns else None
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y_train)

X_val = val[numeric_features + categorical_features].copy()
X_val_trans = preprocessor.transform(X_val)
print(f"Transformed val shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
