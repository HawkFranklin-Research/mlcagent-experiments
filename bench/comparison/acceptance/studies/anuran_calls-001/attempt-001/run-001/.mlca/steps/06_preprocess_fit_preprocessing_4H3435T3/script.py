import mlca_runtime
import json, itertools
import pandas as pd
import numpy as np
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
excluded = protocol['excluded_columns']
if 'group_column' not in protocol:
    raise KeyError(f"group_column missing. Available keys: {list(protocol.keys())}")
group_col = protocol['group_column']

all_cols = list(train.columns)
candidate = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in candidate:
    s = train[c]
    if (s.dtype == object or str(s.dtype) == 'string') or pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s) and len(s) > 1:
            key_like.append(c)
candidate = [c for c in candidate if c not in key_like]
if key_like:
    print(f"Excluded key-like: {key_like}")

excluded_deriv = []
for t in targets:
    t_series = train[t]
    for c in candidate:
        if (train[c] == t_series).all():
            excluded_deriv.append(c)
            print(f"Excluded derivation (exact equality): {c} == {t}")
    if pd.api.types.is_numeric_dtype(t_series):
        numeric_cands = [c for c in candidate if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
        for combo in itertools.chain(itertools.combinations(numeric_cands, 2), itertools.combinations(numeric_cands, 3)):
            if (train[list(combo)].sum(axis=1) == t_series).all():
                for c in combo:
                    if c not in excluded_deriv:
                        excluded_deriv.append(c)
                        print(f"Excluded derivation (sum): {c} part of sum == {t}")
                break
candidate = [c for c in candidate if c not in excluded_deriv]

corr_warn = []
if pd.api.types.is_numeric_dtype(train[targets[0]]):
    for c in candidate:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(train[targets[0]])
            if abs(corr) >= 0.98:
                corr_warn.append(c)
if corr_warn:
    print(f"High correlation (>=0.98) warning: {corr_warn}")

numeric_features, categorical_features = [], []
for c in candidate:
    if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]):
        numeric_features.append(c)
    else:
        cardinality = train[c].nunique()
        if cardinality > 0.05 * len(train):
            print(f"Dropping high-cardinality categorical: {c} ({cardinality} unique)")
        else:
            categorical_features.append(c)

features = numeric_features + categorical_features

placeholders = ['?', '-', 'NA', 'N/A', '']
text_cols = [c for c in features if train[c].dtype == object or str(train[c].dtype) == 'string']
found = []
for c in text_cols:
    s = train[c].astype(str).str.strip()
    mask = s.isin(placeholders) | s.str.upper().isin(['NA','N/A','?','-']) | (s == '')
    if mask.any():
        found.append(c)
if found:
    print(f"Placeholders found in {len(found)} text columns: {found}")
else:
    print("No placeholder tokens found in text feature columns.")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

n_out = len(numeric_features)
for c in categorical_features:
    n_out += train[c].nunique(dropna=True)
if n_out <= 300:
    preprocessor.set_output(transform='pandas')
    numeric_transformer.set_output(transform='pandas')
    categorical_transformer.set_output(transform='pandas')

preprocessor = mlca_runtime.fit_record(preprocessor, train[features], 'train')
val_transformed = preprocessor.transform(val[features])
print(f"Transformed val shape: {val_transformed.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print("Saved preprocessor.pkl")
