import mlca_runtime
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

def get_protocol_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}'. Available keys: {list(d.keys())}")
    return d[key]

targets = get_protocol_key(protocol, 'targets')
excluded = get_protocol_key(protocol, 'excluded_columns')
group_col = get_protocol_key(protocol, 'group_column')
task = get_protocol_key(protocol, 'task')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

# Define feature columns once
drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)
feature_cols = [c for c in train.columns if c not in drop_cols]

# Key-like detection
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == 'object' or s.dtype.name == 'category' or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s) and len(s) > 1:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s) and len(s) > 1:
            if any(k in c.lower() for k in ['id', 'code', 'index', 'key']):
                key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]
if key_like:
    print(f"Dropped key-like columns: {key_like}")

# Screen for exact derivations (categorical target -> skip correlation)
excluded_deriv = []
for c in feature_cols:
    if train[c].equals(train[targets[0]]):
        excluded_deriv.append(c)
        print(f"Excluded feature {c}: exact equality with target")
feature_cols = [c for c in feature_cols if c not in excluded_deriv]

# Sort types
numeric_cols = []
categorical_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# High-cardinality categorical: drop if > 5% unique
n_train = len(train)
high_card = [c for c in categorical_cols if train[c].nunique() > 0.05 * n_train]
if high_card:
    print(f"Dropped high-cardinality categorical columns: {high_card}")
categorical_cols = [c for c in categorical_cols if c not in high_card]

# Placeholder handling
placeholders = ['?', '-', 'NA', 'N/A']
placeholder_cols = []
for c in categorical_cols:
    train[c] = train[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
    val[c] = val[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
    mask = train[c].isin(placeholders) | (train[c] == '')
    if mask.any():
        placeholder_cols.append(c)
        train.loc[mask, c] = np.nan
        val_mask = val[c].isin(placeholders) | (val[c] == '')
        val.loc[val_mask, c] = np.nan
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")

# Build preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

# Set output to pandas if width <= 300
n_out = len(numeric_cols) + sum(train[c].nunique() for c in categorical_cols)
if n_out <= 300:
    numeric_transformer.set_output(transform='pandas')
    categorical_transformer.set_output(transform='pandas')
    preprocessor.set_output(transform='pandas')

# Fit on train
X_train = train[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]
mlca_runtime.fit_record(preprocessor, X_train, 'train', y_train)

# Transform val to check
X_val = val[feature_cols]
X_val_trans = preprocessor.transform(X_val)
print(f"Train shape: {X_train.shape}, Val transformed shape: {X_val_trans.shape}")

# Save
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
