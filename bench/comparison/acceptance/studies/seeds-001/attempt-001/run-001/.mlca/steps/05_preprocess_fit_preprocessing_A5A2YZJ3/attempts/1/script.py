import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing keys {missing}; available: {list(proto.keys())}")

targets = list(proto['targets'])
excluded = list(proto['excluded_columns'])
group_col = proto['group_column']

train = pd.read_parquet(inputs['train'])


def is_key_like(s, name):
    if name in targets or name in excluded or name == group_col:
        return True
    n = len(s)
    if n < 2:
        return False
    nun = s.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(s) and nun == n:
        return True
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and nun == n:
        return True
    return False


candidate = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in candidate if not is_key_like(train[c], c)]

numeric_cols, cat_cols, drop_cols, datetime_cols = [], [], [], []
n_train = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s):
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    else:
        try:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() > 0.9:
                datetime_cols.append(c)
                continue
        except Exception:
            pass
        nun = s.nunique(dropna=True)
        if nun > max(10, int(0.05 * n_train)):
            drop_cols.append(c)
        else:
            cat_cols.append(c)

ph_tokens = ['?', '-', 'NA', 'N/A', '']
ph_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.astype(str).str.strip().isin(ph_tokens).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholder tokens in {len(ph_cols)} column(s)")

log_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 1 and s.min() >= 0 and s.skew() > 1.5:
        log_cols.append(c)
plain_num = [c for c in numeric_cols if c not in log_cols]

print(f"Features: numeric={len(plain_num)}+log={len(log_cols)} cat={len(cat_cols)} dropped={len(drop_cols)}")

if datetime_cols:
    raise RuntimeError("Datetime columns not supported by this preprocessor")

transformers = []
if plain_num:
    transformers.append(('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
    ]), plain_num))
if log_cols:
    transformers.append(('lognum', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('sc', StandardScaler()),
    ]), log_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop')
preprocessor.set_output(transform='pandas')

X_train = train[feature_cols]
fitted = mlca_runtime.fit_record(preprocessor, X_train, 'train')
mlca_runtime.save_model(fitted, 'preprocessor')
mlca_runtime.record_metric('n_features_used', float(len(feature_cols)), 'train')
print(f"Saved preprocessor.pkl with {len(feature_cols)} input features")
