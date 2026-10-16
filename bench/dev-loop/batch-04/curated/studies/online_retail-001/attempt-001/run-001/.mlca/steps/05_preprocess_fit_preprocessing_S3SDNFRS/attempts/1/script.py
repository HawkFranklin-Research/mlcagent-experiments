import mlca_runtime
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
proto = json.loads(inputs['_protocol'].read_text())

targets = proto.get('targets', [])
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')
if group is None:
    raise ValueError(f"Missing 'group_column'. Available keys: {list(proto.keys())}")

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

key_like = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]
print(f"Excluded key-like columns: {key_like}")

numeric_cols = []
categorical_cols = []
date_cols = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_categorical_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        nunique = train[c].nunique()
        if nunique / len(train) > 0.05:
            print(f"Drop high-cardinality text/code column: {c}")
        else:
            categorical_cols.append(c)
    else:
        print(f"Drop unsupported column: {c}")

for c in date_cols:
    dt = pd.to_datetime(train[c])
    for part in ['hour', 'dayofweek', 'month']:
        new_col = f"{c}_{part}"
        train[new_col] = getattr(dt.dt, part)
        numeric_cols.append(new_col)
    print(f"Derived date parts from {c}")

placeholder_cols = []
for c in categorical_cols:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        vals = train[c].dropna().astype(str)
        if vals.str.strip().isin(['?', '-', 'NA', 'N/A', '']).any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholder tokens found in: {placeholder_cols}")

log_cols = []
other_num = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 0 and s.min() >= 0 and s.skew() > 1:
        log_cols.append(c)
    else:
        other_num.append(c)

transformers = []
if log_cols:
    transformers.append(('log_num', Pipeline([
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), log_cols))
if other_num:
    transformers.append(('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), other_num))
if categorical_cols:
    transformers.append(('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
preprocessor.set_output(transform='pandas')

X_train = train[log_cols + other_num + categorical_cols]
preprocessor.fit(X_train)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Fitted preprocessor with {len(log_cols)+len(other_num)} numeric and {len(categorical_cols)} categorical features.")
