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

def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing '{key}' in _protocol. Available keys: {list(d.keys())}")
    return d[key]

targets = get_key(protocol, 'targets')
excluded = get_key(protocol, 'excluded_columns')
group_col = get_key(protocol, 'group_column')
task = get_key(protocol, 'task')
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

target_col = targets[0] if len(targets) == 1 else targets

all_cols = train.columns.tolist()
base_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = []
for c in base_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype.name == 'category':
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholder_tokens).any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholder tokens found in {len(placeholder_cols)} text columns")

feature_cols = []
for c in base_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype.name == 'category':
        if s.nunique() == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            continue
    feature_cols.append(c)

numeric_cols = []
cat_cols = []
datetime_cols = []
drop_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        nunique = s.nunique()
        if nunique > 0.05 * len(s):
            drop_cols.append(c)
        else:
            cat_cols.append(c)

if datetime_cols:
    raise ValueError("Datetime columns not supported in this script; please preprocess separately.")

if drop_cols:
    print(f"Dropped {len(drop_cols)} high-cardinality text columns")

final_features = numeric_cols + cat_cols
X_train = train[final_features]
X_val = val[final_features]

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
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

mlca_runtime.fit_record(preprocessor, X_train, 'train')

X_val_trans = preprocessor.transform(X_val)
print(f"Feature columns: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")
print(f"Train shape: {X_train.shape}, val shape: {X_val.shape}")
print(f"Transformed val shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
