import json
import numpy as np
import pandas as pd
from itertools import combinations
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'; available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

# drop obvious key-like object columns (all distinct text codes)
key_like = [c for c in feature_cols if train[c].dtype == 'object' and train[c].nunique(dropna=True) == len(train)]
feature_cols = [c for c in feature_cols if c not in key_like]

# handle datetime columns: derive numeric parts, drop raw
new_cols = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        dt_train = pd.to_datetime(train[c])
        dt_val = pd.to_datetime(val[c])
        for part in ['year', 'month', 'day', 'dayofweek', 'hour']:
            nc = f"{c}_{part}"
            train[nc] = getattr(dt_train.dt, part)
            val[nc] = getattr(dt_val.dt, part)
            new_cols.append(nc)
    else:
        new_cols.append(c)
feature_cols = new_cols

# text placeholders
placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feature_cols:
    if train[c].dtype == 'object' or pd.api.types.is_string_dtype(train[c]):
        train[c] = train[c].astype(str).str.strip()
        val[c] = val[c].astype(str).str.strip()
        mask = train[c].isin(placeholders)
        if mask.any():
            ph_cols.append(c)
            train.loc[mask, c] = np.nan
            val[c] = val[c].replace(list(placeholders), np.nan)
if ph_cols:
    print(f"Placeholders found in {len(ph_cols)} text columns: {ph_cols}")

# drop high-cardinality text (>5% unique)
high_card = [c for c in feature_cols if train[c].dtype == 'object' and train[c].nunique(dropna=True) > 0.05 * len(train)]
feature_cols = [c for c in feature_cols if c not in high_card]
if high_card:
    print(f"Dropped high-cardinality text: {high_card}")

# exact derivation screening
target_col = targets[0]
y_train = train[target_col]
derived = []
for c in feature_cols:
    if train[c].dtype.kind in 'if' and np.array_equal(train[c].values, y_train.values):
        derived.append(c)
nonneg = [c for c in feature_cols if train[c].dtype.kind in 'if' and (train[c] >= 0).all()]
for r in range(1, 4):
    for combo in combinations(nonneg, r):
        if any(c in derived for c in combo):
            continue
        s = train[list(combo)].sum(axis=1)
        if np.array_equal(s.values, y_train.values):
            derived.extend([c for c in combo if c not in derived])
feature_cols = [c for c in feature_cols if c not in derived]
for c in derived:
    print(f"Excluded derived feature: {c}")

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    if pd.api.types.is_bool_dtype(train[c]):
        categorical_cols.append(c)
    elif train[c].dtype.kind in 'if':
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, categorical_cols)
])

expected_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in categorical_cols)
if expected_width <= 300:
    for obj in (preprocessor, numeric_pipe, categorical_pipe):
        obj.set_output(transform='pandas')

X_train = train[feature_cols]
fitted = mlca_runtime.fit_record(preprocessor, X_train, 'train', y_train)
X_val_trans = fitted.transform(val[feature_cols])
print(f"Transformed val shape: {X_val_trans.shape}")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
