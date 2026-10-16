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
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

def get_key(d, key, default=None):
    if key in d:
        return d[key]
    if default is not None:
        return default
    raise KeyError(f"Missing key '{key}'. Available: {list(d.keys())}")

targets = get_key(protocol, 'targets')
if isinstance(targets, str):
    targets = [targets]
excluded = get_key(protocol, 'excluded_columns', [])
group = get_key(protocol, 'group_column', None)
task = get_key(protocol, 'task', {})

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

# Key-like detection
key_like = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# Placeholder handling
placeholders = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        mask = s.isin(placeholders) | s.str.match(r'^\s*$')
        if mask.any():
            placeholder_cols.append(c)
            train[c] = train[c].astype(str).str.strip().replace(placeholders, np.nan)
            val[c] = val[c].astype(str).str.strip().replace(placeholders, np.nan)

# Date/time detection
date_cols = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[c]):
        try:
            pd.to_datetime(train[c].head(100), errors='raise')
            date_cols.append(c)
        except Exception:
            pass
for c in date_cols:
    dt = pd.to_datetime(train[c], errors='coerce')
    train[c+'_hour'] = dt.dt.hour
    train[c+'_dow'] = dt.dt.dayofweek
    train[c+'_month'] = dt.dt.month
    val_dt = pd.to_datetime(val[c], errors='coerce')
    val[c+'_hour'] = val_dt.dt.hour
    val[c+'_dow'] = val_dt.dt.dayofweek
    val[c+'_month'] = val_dt.dt.month
    feature_cols.remove(c)
    feature_cols.extend([c+'_hour', c+'_dow', c+'_month'])

# High-cardinality categorical
n = len(train)
high_card = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[c]):
        if train[c].nunique() > 0.05 * n:
            high_card.append(c)
feature_cols = [c for c in feature_cols if c not in high_card]

# Exclude exact derivations of target
y_train = train[targets[0]] if targets else None
derived = []
if y_train is not None:
    for c in feature_cols:
        if train[c].equals(y_train):
            derived.append(c)
feature_cols = [c for c in feature_cols if c not in derived]

# Split numeric and categorical
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

# Build preprocessor
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
transformers = []
if numeric_cols:
    transformers.append(('num', numeric_transformer, numeric_cols))
if categorical_cols:
    transformers.append(('cat', categorical_transformer, categorical_cols))
preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

fitted_preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y_train)

# Set output pandas if <= 300 columns
try:
    width = len(fitted_preprocessor.get_feature_names_out())
    if width <= 300:
        fitted_preprocessor.set_output(transform='pandas')
        for name, trans, cols in fitted_preprocessor.transformers_:
            if hasattr(trans, 'set_output'):
                trans.set_output(transform='pandas')
except Exception:
    pass

val_transformed = fitted_preprocessor.transform(X_val)

# Print summary (max 5 lines)
print(f"Train: {train.shape}, Val: {val.shape}")
print(f"Features: {len(feature_cols)} (num: {len(numeric_cols)}, cat: {len(categorical_cols)})")
print(f"Dropped: key-like={key_like}, high-card={high_card}, derived={derived}")
if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} cols: {placeholder_cols}")
if date_cols:
    print(f"Parsed dates: {date_cols}")

mlca_runtime.save_model(fitted_preprocessor, "preprocessor.pkl")
