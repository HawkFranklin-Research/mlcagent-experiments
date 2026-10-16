import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing key {k}; available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = [c for c in train.columns if c not in drop]

def is_keylike(s):
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

keylike = [c for c in feature_cols if is_keylike(train[c])]
feature_cols = [c for c in feature_cols if c not in keylike]

placeholder_cols = []
for c in feature_cols:
    if train[c].dtype == object:
        s = train[c].astype(str).str.strip()
        if s.isin(['?', '-', 'NA', 'N/A', '']).any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns")

numeric_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        nun = train[c].nunique(dropna=True)
        if nun <= max(1, int(0.05 * len(train))):
            cat_cols.append(c)

numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
numeric_pipe.set_output(transform='pandas')

transformers = [('num', numeric_pipe, numeric_cols)]
if cat_cols:
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    cat_pipe.set_output(transform='pandas')
    transformers.append(('cat', cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder='drop')
pre.set_output(transform='pandas')

X_train = train[feature_cols]
mlca_runtime.fit_record(pre, X_train, 'train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')
print(f"Fitted preprocessor: {len(feature_cols)} features ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)")
