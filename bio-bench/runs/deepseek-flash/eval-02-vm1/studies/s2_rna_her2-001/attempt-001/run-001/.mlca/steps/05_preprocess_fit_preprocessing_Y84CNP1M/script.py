import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise ValueError(f"Missing '{key}' in protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = set(protocol['excluded_columns']) if isinstance(protocol['excluded_columns'], list) else set()
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

y_train = train[targets[0]]

# candidate features
feat_cols = [c for c in train.columns if c not in set(targets) | excluded and c != group]
# drop key-like text columns (all distinct values)
for c in list(feat_cols):
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train):
            feat_cols.remove(c)

# screen exact target derivations (binary target: exact equality, or sum of up to 3 binary cols)
excluded_deriv = []
binary_cols = []
for c in feat_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        if train[c].equals(y_train):
            excluded_deriv.append((c, 'exact equality with target'))
        elif set(train[c].dropna().unique()).issubset({0, 1}):
            binary_cols.append(c)

# check pairs/triples of binary columns summing to target (only if target is binary)
if set(y_train.dropna().unique()).issubset({0, 1}):
    for i, c1 in enumerate(binary_cols):
        if c1 in [x[0] for x in excluded_deriv]:
            continue
        for c2 in binary_cols[i+1:]:
            if c2 in [x[0] for x in excluded_deriv]:
                continue
            if (train[c1] + train[c2]).equals(y_train):
                excluded_deriv.append((c1, 'sum of 2 binary cols equals target'))
                excluded_deriv.append((c2, 'sum of 2 binary cols equals target'))
                break

feat_cols = [c for c in feat_cols if c not in [x[0] for x in excluded_deriv]]

# split types, handle datetime
numeric_feats = []
categorical_feats = []
for c in feat_cols:
    dt = train[c].dtype
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for part, func in [('hour', lambda s: s.dt.hour), ('dow', lambda s: s.dt.dayofweek), ('month', lambda s: s.dt.month)]:
            new_name = f"{c}_{part}"
            train[new_name] = func(train[c])
            val[new_name] = func(val[c])
            numeric_feats.append(new_name)
        continue
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_feats.append(c)
    else:
        card = train[c].nunique()
        if card <= 0.05 * len(train):
            categorical_feats.append(c)
        # else drop high-cardinality text

final_feats = numeric_feats + categorical_feats

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ],
    remainder='drop'
)

est_width = len(numeric_feats) + sum(train[c].nunique() for c in categorical_feats)
if est_width <= 300:
    preprocessor.set_output(transform='pandas')

mlca_runtime.fit_record(preprocessor, train[final_feats], 'train', y=y_train)

X_val_trans = preprocessor.transform(val[final_feats])

print(f"Features: {len(numeric_feats)} numeric, {len(categorical_feats)} categorical")
print(f"Excluded derivations: {len(excluded_deriv)}")
print(f"Train transformed shape: {preprocessor.transform(train[final_feats]).shape}")
print(f"Val transformed shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
