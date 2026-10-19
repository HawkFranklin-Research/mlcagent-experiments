import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn import set_config

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

key_like = []
for c in feature_cols:
    if train[c].dtype == object or pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

datetime_cols = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        datetime_cols.append(c)
    elif train[c].dtype == object:
        sample = train[c].dropna().head(100)
        try:
            pd.to_datetime(sample)
            datetime_cols.append(c)
        except Exception:
            pass
feature_cols = [c for c in feature_cols if c not in datetime_cols]

high_card = []
for c in feature_cols:
    if train[c].dtype == object:
        if train[c].nunique() > 0.05 * len(train):
            high_card.append(c)
feature_cols = [c for c in feature_cols if c not in high_card]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

log_cols = []
for c in numeric_cols:
    if train[c].min() >= 0 and abs(train[c].skew()) > 1:
        log_cols.append(c)
numeric_no_log = [c for c in numeric_cols if c not in log_cols]

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
log_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

transformers = []
if numeric_no_log:
    transformers.append(('num', numeric_pipeline, numeric_no_log))
if log_cols:
    transformers.append(('log', log_pipeline, log_cols))
if categorical_cols:
    transformers.append(('cat', categorical_pipeline, categorical_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop')
preprocessor.fit(train)

try:
    if len(preprocessor.get_feature_names_out()) <= 300:
        set_config(transform_output="pandas")
        preprocessor.set_output(transform="pandas")
except Exception:
    pass

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Features: {len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"Excluded datetime: {datetime_cols}")
print(f"Excluded high-card: {high_card}")
print(f"Log1p applied: {log_cols}")
