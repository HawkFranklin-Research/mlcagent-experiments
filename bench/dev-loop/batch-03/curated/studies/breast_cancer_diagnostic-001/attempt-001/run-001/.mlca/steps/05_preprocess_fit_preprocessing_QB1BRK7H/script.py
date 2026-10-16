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
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

# Drop key-like columns
key_like = []
for c in feature_cols[:]:
    if train[c].dtype == 'object' or train[c].dtype.name == 'category':
        if train[c].nunique() == len(train):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
for c in key_like:
    feature_cols.remove(c)
if key_like:
    print(f"Dropped key-like: {key_like}")

# Screen target derivations
target = targets[0]
excluded_deriv = []
for c in feature_cols[:]:
    if train[c].astype(str).equals(train[target].astype(str)):
        excluded_deriv.append(c)
        feature_cols.remove(c)
if excluded_deriv:
    print(f"Excluded derivations: {excluded_deriv}")

# Sort types
numeric_cols = []
cat_cols = []
high_card = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        if train[c].nunique() > 0.05 * len(train):
            high_card.append(c)
        else:
            cat_cols.append(c)
for c in high_card:
    feature_cols.remove(c)
if high_card:
    print(f"Dropped high-cardinality: {high_card}")

# Check placeholders
placeholders = ['?', '-', 'NA', 'N/A']
found_placeholders = set()
placeholder_cols = []
for c in cat_cols:
    vals = train[c].dropna().astype(str)
    for p in placeholders:
        if (vals == p).any():
            found_placeholders.add(p)
            placeholder_cols.append(c)
            break
if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} cols: {placeholder_cols}")
else:
    print("No placeholders found.")

# Build preprocessor
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_steps = []
for p in found_placeholders:
    cat_steps.append((f'imputer_{p}', SimpleImputer(missing_values=p, strategy='most_frequent')))
cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
categorical_transformer = Pipeline(cat_steps)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

width = len(numeric_cols) + sum(train[c].nunique() for c in cat_cols)
if width <= 300:
    preprocessor.set_output(transform='pandas')
    numeric_transformer.set_output(transform='pandas')
    categorical_transformer.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[target]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y=y_train)

X_val = val[feature_cols]
X_val_trans = preprocessor.transform(X_val)
print(f"Train shape: {preprocessor.transform(X_train).shape}, Val shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric('n_features', len(feature_cols), 'train')
mlca_runtime.record_metric('n_train', len(train), 'train')
mlca_runtime.record_metric('n_val', len(val), 'val')
