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
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

# key-like exclusion: object columns with all distinct values
key_like = [c for c in feature_cols if train[c].dtype == 'object' and train[c].nunique() == len(train) and train[c].nunique() > 1]
feature_cols = [c for c in feature_cols if c not in key_like]

# exact target derivation screen
excluded_features = []
if len(targets) == 1:
    y_train = train[targets[0]]
    for c in feature_cols:
        if np.allclose(train[c].values, y_train.values, equal_nan=True):
            excluded_features.append(c)
feature_cols = [c for c in feature_cols if c not in excluded_features]

# placeholder check (report only)
placeholder_cols = []
for c in feature_cols:
    if train[c].dtype == 'object':
        s = train[c].astype(str)
        mask = s.isin(['?', '-', 'NA', 'N/A']) | s.str.strip().eq('')
        if mask.any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")
else:
    print("No placeholder tokens found.")

# sort feature types
numeric_features = []
categorical_features = []
datetime_features = []
dropped_high_card = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_features.append(c)
    elif pd.api.types.is_datetime64_any_dtype(train[c]):
        datetime_features.append(c)
    elif train[c].dtype == 'object' or isinstance(train[c].dtype, pd.CategoricalDtype):
        if train[c].nunique() <= 0.05 * len(train):
            categorical_features.append(c)
        else:
            dropped_high_card.append(c)
feature_cols = [c for c in feature_cols if c not in dropped_high_card]
if dropped_high_card:
    print(f"Dropped high-cardinality text columns: {dropped_high_card}")

# derive datetime parts and drop raw
for c in datetime_features:
    dt_train = pd.to_datetime(train[c])
    dt_val = pd.to_datetime(val[c])
    for suf, func in [('_hour', lambda d: d.dt.hour), ('_dow', lambda d: d.dt.dayofweek), ('_month', lambda d: d.dt.month)]:
        new_col = c + suf
        train[new_col] = func(dt_train)
        val[new_col] = func(dt_val)
        numeric_features.append(new_col)
feature_cols = numeric_features + categorical_features

# build ColumnTransformer
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
if categorical_features:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
else:
    transformers = [('num', numeric_transformer, numeric_features)]

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

# estimate output width for pandas output
n_numeric = len(numeric_features)
n_cat = sum(train[c].nunique() for c in categorical_features) if categorical_features else 0
if n_numeric + n_cat <= 300:
    preprocessor.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[targets[0]] if targets else None
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y=y_train)

X_val = val[feature_cols]
X_val_trans = preprocessor.transform(X_val)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Transformed val: {X_val_trans.shape}")

if excluded_features:
    print(f"Excluded features as target derivations: {excluded_features}")
else:
    print("No features excluded as target derivations.")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
