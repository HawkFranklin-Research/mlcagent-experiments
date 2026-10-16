import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol.get('targets')
if targets is None:
    raise ValueError(f"Missing targets. Available keys: {list(protocol.keys())}")
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
task = protocol.get('task', {})

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

msgs = []

for c in list(feature_cols):
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique(dropna=False) == len(train):
            feature_cols.remove(c)
            msgs.append(f"Excluded key-like column: {c}")

placeholders = ['?', '-', 'NA', 'N/A', '']
ph_cols = []
for c in feature_cols:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        s = train[c].dropna().astype(str)
        mask = s.str.strip().isin(placeholders) | (s != s.str.strip())
        if mask.any():
            ph_cols.append(c)
if ph_cols:
    msgs.append(f"Placeholders found in {len(ph_cols)} columns: {ph_cols[:5]}")

for df in [train, val]:
    for c in feature_cols:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
            df[c] = df[c].replace(placeholders, np.nan)

y = train[targets[0]] if targets else None
if y is not None:
    for c in list(feature_cols):
        if train[c].equals(y):
            feature_cols.remove(c)
            msgs.append(f"Excluded exact target derivation: {c}")
    num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(num_feats, r):
            if (train[list(combo)].sum(axis=1) == y).all():
                for c in combo:
                    if c in feature_cols:
                        feature_cols.remove(c)
                        msgs.append(f"Excluded sum derivation: {c}")

if y is not None and pd.api.types.is_numeric_dtype(y):
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if abs(corr) >= 0.98:
                msgs.append(f"Warning: high correlation {corr:.2f} for {c}")

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(train[c]):
        for part in ['hour', 'dayofweek', 'month']:
            new_c = f"{c}_{part}"
            train[new_c] = getattr(train[c].dt, part)
            val[new_c] = getattr(val[c].dt, part)
            numeric_cols.append(new_c)
    else:
        if train[c].nunique() > 0.05 * len(train):
            msgs.append(f"Dropped high-cardinality: {c}")
        else:
            categorical_cols.append(c)

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
])
preprocessor.set_output(transform='pandas')

X_train = train[numeric_cols + categorical_cols]
mlca_runtime.fit_record(preprocessor, X_train, 'train', y=y)
X_val = val[numeric_cols + categorical_cols]
preprocessor.transform(X_val)

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')

for m in msgs[:4]:
    print(m)
print(f"X_train shape: {X_train.shape}")
