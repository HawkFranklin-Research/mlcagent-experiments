import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import itertools
import warnings
warnings.filterwarnings('ignore')

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing key {key} in _protocol. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = train.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

# Drop key-like columns (all distinct text or integer sequences, not 0/1 indicators)
key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s) and not pd.api.types.is_numeric_dtype(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.nunique() == len(s) and not (s.isin([0, 1]).all()):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# Text placeholder check
placeholder_cols = []
placeholders = ['?', '-', 'NA', 'N/A', '']
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        mask = s.isin(placeholders) | (s == '')
        if mask.any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholder tokens found in {len(placeholder_cols)} columns: {placeholder_cols[:5]}...")
else:
    print("No text placeholder tokens found in feature columns.")

# Target derivation screening (exact equality or sums of up to 3 binary features)
y_train = train[targets[0]].values
excluded_deriv = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        col = train[c].values
        if np.allclose(col, y_train, equal_nan=True):
            excluded_deriv.append((c, 'exact equality with target'))

if np.issubdtype(y_train.dtype, np.number):
    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    corrs = {}
    for c in numeric_feats:
        s = train[c]
        if s.std() > 0:
            corrs[c] = abs(np.corrcoef(s, y_train)[0, 1])
    high_corr = [c for c, v in corrs.items() if v >= 0.98]
    binary_feats = [c for c in high_corr if (train[c] >= 0).all() and train[c].nunique() <= 2]
    for r in [2, 3]:
        for combo in itertools.combinations(binary_feats, r):
            s = train[list(combo)].sum(axis=1)
            if np.allclose(s, y_train, equal_nan=True):
                for c in combo:
                    excluded_deriv.append((c, f'sum with {combo} equals target'))
                break
    excluded_deriv = list(set(excluded_deriv))
    if excluded_deriv:
        for c, reason in excluded_deriv:
            print(f"Excluded feature {c}: {reason}")
        feature_cols = [c for c in feature_cols if c not in [x[0] for x in excluded_deriv]]
    remaining_high = [c for c in high_corr if c in feature_cols]
    if remaining_high:
        print(f"Warning: features with |corr| >= 0.98 with target: {remaining_high[:5]}...")

# Sort feature columns into numeric, low-cardinality categorical
numeric_cols = []
categorical_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique() <= 0.05 * len(train):
            categorical_cols.append(c)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
if categorical_cols:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
else:
    categorical_transformer = 'passthrough'

n_numeric = len(numeric_cols)
n_cat = sum(train[c].nunique() for c in categorical_cols) if categorical_cols else 0
total_width = n_numeric + n_cat
if total_width > 300:
    if categorical_cols:
        categorical_transformer.set_params(onehot__sparse_output=True)
    set_output = False
else:
    set_output = True

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)
if set_output:
    preprocessor.set_output(transform='pandas')

input_cols = numeric_cols + categorical_cols
mlca_runtime.fit_record(preprocessor, train[input_cols], partition='train', y=y_train)
val_transformed = preprocessor.transform(val[input_cols])
print(f"Val transformed shape: {val_transformed.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
