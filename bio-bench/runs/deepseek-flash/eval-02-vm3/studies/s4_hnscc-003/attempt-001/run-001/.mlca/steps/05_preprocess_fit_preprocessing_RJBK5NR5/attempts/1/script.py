import mlca_runtime
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import itertools

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

key_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == 'object' or s.dtype.name == 'category':
        if s.nunique() == len(s) and not s.str.contains(r'\d').any():
            key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and s.nunique() > 10:
        key_cols.append(c)
for c in key_cols:
    print(f"Excluding key-like column: {c}")
feature_cols = [c for c in feature_cols if c not in key_cols]

excluded_features = []
num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for target in targets:
    if target not in train.columns:
        continue
    t = train[target]
    for c in feature_cols[:]:
        if c in excluded_features:
            continue
        s = train[c]
        if s.equals(t):
            excluded_features.append(c)
            print(f"Excluding {c}: exact match with target {target}")
            continue
    if not pd.api.types.is_numeric_dtype(t):
        continue
    for combo in itertools.combinations(num_feats, 2):
        if any(c in excluded_features for c in combo):
            continue
        if (train[list(combo)].sum(axis=1) == t).all():
            for c in combo:
                if c not in excluded_features:
                    excluded_features.append(c)
                    print(f"Excluding {c}: sum of {combo} equals target {target}")
    if len(num_feats) < 20:
        for combo in itertools.combinations(num_feats, 3):
            if any(c in excluded_features for c in combo):
                continue
            if (train[list(combo)].sum(axis=1) == t).all():
                for c in combo:
                    if c not in excluded_features:
                        excluded_features.append(c)
                        print(f"Excluding {c}: sum of {combo} equals target {target}")

feature_cols = [c for c in feature_cols if c not in excluded_features]

for target in targets:
    if target not in train.columns or not pd.api.types.is_numeric_dtype(train[target]):
        continue
    t = train[target].astype(float)
    numeric_cols_temp = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    high_corr = []
    for c in numeric_cols_temp:
        corr = train[c].astype(float).corr(t)
        if abs(corr) >= 0.98:
            high_corr.append((c, corr))
    if high_corr:
        names = ', '.join([f"{c} ({corr:.4f})" for c, corr in high_corr])
        print(f"Warning: high correlation with {target}: {names}")

text_cols = [c for c in feature_cols if train[c].dtype == 'object']
placeholder_tokens = ['?', '-', 'NA', 'N/A']
placeholder_found = []
for c in text_cols:
    s = train[c].astype(str).str.strip()
    for tok in placeholder_tokens:
        if (s == tok).any():
            placeholder_found.append(c)
            break
if placeholder_found:
    print(f"Placeholders found in {len(placeholder_found)} columns: {placeholder_found}")

numeric_cols = []
categorical_cols = []
datetime_cols = []
drop_cols = []
n = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.dtype == 'object' or s.dtype.name == 'category':
        if s.nunique() > 0.05 * n:
            drop_cols.append(c)
        else:
            categorical_cols.append(c)

for c in datetime_cols:
    dt = pd.to_datetime(train[c], errors='coerce')
    train[f'{c}_hour'] = dt.dt.hour
    train[f'{c}_dayofweek'] = dt.dt.dayofweek
    train[f'{c}_month'] = dt.dt.month
    val_dt = pd.to_datetime(val[c], errors='coerce')
    val[f'{c}_hour'] = val_dt.dt.hour
    val[f'{c}_dayofweek'] = val_dt.dt.dayofweek
    val[f'{c}_month'] = val_dt.dt.month
    numeric_cols.extend([f'{c}_hour', f'{c}_dayofweek', f'{c}_month'])

if drop_cols:
    print(f"Dropping {len(drop_cols)} high-cardinality text columns")

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
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

X_train = train[numeric_cols + categorical_cols]
X_val = val[numeric_cols + categorical_cols]

approx_width = len(numeric_cols) + sum(train[c].nunique() for c in categorical_cols)
if approx_width <= 300:
    preprocessor.set_output(transform='pandas')

mlca_runtime.fit_record(preprocessor, X_train, 'train')
X_val_trans = preprocessor.transform(X_val)
print(f"Train shape: {X_train.shape}, Val transformed shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
