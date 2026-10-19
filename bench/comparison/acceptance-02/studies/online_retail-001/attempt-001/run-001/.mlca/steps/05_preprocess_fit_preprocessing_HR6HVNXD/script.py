import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

key_like = []
for c in feature_cols:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
        if s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

dt_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_object_dtype(s):
        sample = s.dropna().head(50)
        if len(sample):
            try:
                pd.to_datetime(sample, errors='raise')
                dt_cols.append(c)
            except Exception:
                pass
feature_cols = [c for c in feature_cols if c not in dt_cols]

high_card = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]):
        if train[c].nunique(dropna=True) > 0.05 * len(train):
            high_card.append(c)
feature_cols = [c for c in feature_cols if c not in high_card]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

log_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) and s.min() >= 0 and abs(s.skew()) > 1:
        log_cols.append(c)
plain_num = [c for c in numeric_cols if c not in log_cols]

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
log_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

transformers = []
if plain_num:
    transformers.append(('num', num_pipe, plain_num))
if log_cols:
    transformers.append(('log', log_pipe, log_cols))
if categorical_cols:
    transformers.append(('cat', cat_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop')

X_train = train[feature_cols].copy()
mlca_runtime.fit_record(preprocessor, X_train, 'train')

try:
    if len(preprocessor.get_feature_names_out()) <= 300:
        preprocessor.set_output(transform='pandas')
except Exception:
    pass

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]):
        vals = train[c].dropna().astype(str).str.strip()
        if vals.isin(ph_tokens).any():
            ph_cols.append(c)
if ph_cols:
    print(f"placeholder tokens in {len(ph_cols)} cols: {ph_cols}")

print(f"features={len(feature_cols)} num={len(plain_num)} log={len(log_cols)} cat={len(categorical_cols)}")
if key_like:
    print(f"key-like dropped: {key_like}")
if dt_cols:
    print(f"datetime dropped: {dt_cols}")
if high_card:
    print(f"high-card dropped: {high_card}")
