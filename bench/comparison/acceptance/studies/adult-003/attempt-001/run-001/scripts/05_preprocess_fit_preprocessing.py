import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column', None)
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
candidate = [c for c in all_cols if c not in targets and c not in excluded and c != group]

key_like = []
for c in candidate:
    if pd.api.types.is_string_dtype(train[c]) or train[c].dtype == 'object':
        if train[c].nunique() == len(train):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train):
        if 'id' in c.lower() or 'key' in c.lower():
            key_like.append(c)
candidate = [c for c in candidate if c not in key_like]

derived = {}
for c in candidate:
    for t in targets:
        if (train[c] == train[t]).all():
            derived[c] = f"exact equality with {t}"
            break
for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]) and not pd.api.types.is_bool_dtype(train[t]):
        nonneg = [c for c in candidate if c not in derived and (train[c] >= 0).all()]
        for r in range(1, 4):
            for combo in combinations(nonneg, r):
                if (train[list(combo)].sum(axis=1) == train[t]).all():
                    for c in combo:
                        derived[c] = f"sum {combo} equals {t}"
feature_cols = [c for c in candidate if c not in derived]

placeholders = ['?', '-', 'NA', 'N/A']
ph_cols = []
for c in feature_cols:
    if pd.api.types.is_string_dtype(train[c]) or train[c].dtype == 'object':
        s = train[c].astype(str)
        if s.isin(placeholders).any() or s.str.strip().eq('').any():
            ph_cols.append(c)

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]
low_card = []
dropped_high = []
for c in categorical_cols:
    if train[c].nunique() > 0.05 * len(train):
        dropped_high.append(c)
    else:
        low_card.append(c)
categorical_cols = low_card
feature_cols = numeric_cols + categorical_cols

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
cat_pipe = Pipeline([
    ('strip', FunctionTransformer(pd.DataFrame.replace, kw_args={'to_replace': r'^\s+|\s+$', 'value': '', 'regex': True})),
    ('replace_ph', FunctionTransformer(pd.DataFrame.replace, kw_args={'to_replace': placeholders + [''], 'value': np.nan})),
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
cat_pipe.set_output(transform='pandas')
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, categorical_cols)
])
preprocessor.set_output(transform='pandas')

X_train = train[feature_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train')
X_val_trans = preprocessor.transform(val[feature_cols])

if key_like:
    print(f"Excluded key-like: {key_like}")
if derived:
    print(f"Excluded derived: {list(derived.keys())}")
if ph_cols:
    print(f"Placeholders in {len(ph_cols)} columns")
if dropped_high:
    print(f"Dropped high-cardinality: {dropped_high}")
print(f"Features: {len(feature_cols)}, Val transformed: {X_val_trans.shape}")
mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
