import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from itertools import combinations

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

for k in ['targets', 'excluded_columns']:
    if k not in protocol:
        raise KeyError(f"Missing {k} in _protocol. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

logs = []
candidates = [c for c in train.columns if c not in targets and c not in excluded]
if group_col and group_col in candidates:
    candidates.remove(group_col)

# drop key-like text columns
for col in candidates[:]:
    if pd.api.types.is_string_dtype(train[col]) or train[col].dtype == 'object':
        if train[col].nunique(dropna=False) == len(train):
            logs.append(f"Excluded identifier column: {col}")
            candidates.remove(col)

# placeholder check
placeholder_tokens = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = []
for col in candidates:
    if pd.api.types.is_string_dtype(train[col]) or train[col].dtype == 'object':
        s = train[col].astype(str).str.strip()
        for tok in placeholder_tokens:
            if (s == tok).any():
                placeholder_cols.append(col)
                break
logs.append(f"Placeholders found in {len(placeholder_cols)} columns" if placeholder_cols else "No placeholder tokens found in text columns")

# screen derivations
excluded_features = set()
for target in targets:
    if target not in train: continue
    y = train[target]
    if not pd.api.types.is_numeric_dtype(y): continue
    for feat in candidates:
        if feat in excluded_features: continue
        if pd.api.types.is_numeric_dtype(train[feat]) and (train[feat] == y).all():
            excluded_features.add(feat)
    eligible = [f for f in candidates if f not in excluded_features and pd.api.types.is_numeric_dtype(train[f]) and train[f].min() >= 0 and train[f].max() <= 1]
    for r in [1,2,3]:
        for combo in combinations(eligible, r):
            if all(f not in excluded_features for f in combo):
                if (train[list(combo)].sum(axis=1) == y).all():
                    for f in combo:
                        excluded_features.add(f)

if excluded_features:
    logs.append(f"Excluded derivation features: {sorted(excluded_features)}")

final_features = [f for f in candidates if f not in excluded_features]

# correlation warning
warn_features = set()
for target in targets:
    if target not in train: continue
    y = train[target]
    if not pd.api.types.is_numeric_dtype(y): continue
    for feat in final_features:
        if pd.api.types.is_numeric_dtype(train[feat]):
            corr = train[feat].corr(y)
            if abs(corr) >= 0.98:
                warn_features.add(feat)
if warn_features:
    logs.append(f"Correlation >=0.98 with target: {sorted(warn_features)}")

# type sorting
numeric_cols = []
cat_cols = []
for col in final_features:
    if pd.api.types.is_datetime64_any_dtype(train[col]):
        logs.append(f"Dropped datetime column {col}")
    elif pd.api.types.is_numeric_dtype(train[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_string_dtype(train[col]) or train[col].dtype == 'object':
        nunique = train[col].nunique()
        if nunique <= max(5, 0.05 * len(train)):
            cat_cols.append(col)
        else:
            logs.append(f"Dropped high-cardinality column {col} (nunique={nunique})")
    else:
        logs.append(f"Dropped unsupported column {col} dtype={train[col].dtype}")

width = len(numeric_cols) + sum(train[c].nunique() for c in cat_cols)
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
if width <= 300:
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipe.set_output(transform='pandas')
    cat_pipe.set_output(transform='pandas')
    ct = ColumnTransformer(transformers=[])
    ct.set_output(transform='pandas')
else:
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    ct = ColumnTransformer(transformers=[], sparse_threshold=1.0)

transformers = []
if numeric_cols:
    transformers.append(('num', num_pipe, numeric_cols))
if cat_cols:
    transformers.append(('cat', cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder='drop')
if width <= 300:
    preprocessor.set_output(transform='pandas')

X_train = train[final_features]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train')
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
logs.append(f"Fitted preprocessor on {len(final_features)} features, saved preprocessor.pkl")

for line in logs[:5]:
    print(line)
