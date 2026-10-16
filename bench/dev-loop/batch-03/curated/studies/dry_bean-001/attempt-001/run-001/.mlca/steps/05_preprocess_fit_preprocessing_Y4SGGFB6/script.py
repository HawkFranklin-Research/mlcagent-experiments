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

def get_required(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing required key '{key}' in protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_required(protocol, 'targets')
excluded = get_required(protocol, 'excluded_columns')
group_col = protocol.get('group_column')
task = get_required(protocol, 'task')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(series):
    if series.dtype == 'object' or series.dtype.name == 'category':
        return series.nunique() == len(series)
    if pd.api.types.is_integer_dtype(series):
        return series.nunique() == len(series)
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])]

excluded_features = []
for col in feature_cols:
    if train[col].dtype == train[targets[0]].dtype:
        if train[col].equals(train[targets[0]]):
            excluded_features.append((col, 'exact equality with target'))
feature_cols = [c for c in feature_cols if c not in [e[0] for e in excluded_features]]
print(f"Excluded features: {excluded_features if excluded_features else 'none'}")

placeholders = ['?', '-', 'NA', 'N/A']
placeholder_cols = []
for col in feature_cols:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        s = train[col].dropna().astype(str).str.strip()
        if s.isin(placeholders + ['']).any():
            placeholder_cols.append(col)
print(f"Placeholder columns: {len(placeholder_cols)}")

numeric_cols = []
categorical_cols = []
text_cols = []
date_cols = []
n = len(train)
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(train[col]):
        date_cols.append(col)
    else:
        nunique = train[col].nunique()
        if nunique <= 0.05 * n:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='drop')
preprocessor.set_output(transform='pandas')
numeric_transformer.set_output(transform='pandas')
categorical_transformer.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[targets[0]]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y_train)

X_val = val[feature_cols]
X_val_trans = preprocessor.transform(X_val)
print(f"Train shape: {X_train.shape}, Val transformed shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
