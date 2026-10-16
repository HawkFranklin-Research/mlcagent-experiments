import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
protocol = json.loads(inputs['_protocol'].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing _protocol key '{k}'. Available: {list(d.keys())}")
    return d[k]

targets = req(protocol, 'targets')
excluded = req(protocol, 'excluded_columns')
group = req(protocol, 'group_column')
task = req(protocol, 'task')

all_cols = list(train.columns)
cand = [c for c in all_cols if c not in targets and c not in excluded and c != group]
feature_cols = []
for c in cand:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=False) == len(train):
        print(f"drop identifier {c}")
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(train):
        print(f"drop identifier {c}")
        continue
    feature_cols.append(c)

ph_cols = 0
for c in feature_cols:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c].dtype):
        s = train[c].astype(str).str.strip()
        if any((s == tok).any() for tok in ['?', '-', 'NA', 'N/A', '']):
            ph_cols += 1
print(f"placeholders: {ph_cols} columns")

num_cols, cat_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        try:
            pd.to_datetime(s, errors='raise')
            continue
        except Exception:
            if s.nunique(dropna=False) > 0.05 * len(train):
                high_card.append(c)
            else:
                cat_cols.append(c)
    else:
        num_cols.append(c)

log_cols = [c for c in num_cols if train[c].dropna().min() >= 0 and train[c].dropna().skew() > 1.0]
regular_num = [c for c in num_cols if c not in log_cols]
print(f"log1p: {log_cols if log_cols else 'none'}")
print(f"features: {len(feature_cols)} total, {len(regular_num)} num, {len(log_cols)} log1p, {len(cat_cols)} cat, {len(high_card)} high-card dropped")

width = len(regular_num) + len(log_cols)
for c in cat_cols:
    width += train[c].nunique(dropna=False)
use_sparse = width > 300
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=use_sparse)

transformers = []
if regular_num:
    transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), regular_num))
if log_cols:
    transformers.append(('log', Pipeline([('imputer', SimpleImputer(strategy='median')), ('log', FunctionTransformer(np.log1p, validate=False)), ('scaler', StandardScaler())]), log_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', onehot)]), cat_cols))

pre = ColumnTransformer(transformers, remainder='drop')
if not use_sparse:
    pre.set_output(transform='pandas')

mlca_runtime.fit_record(pre, train[feature_cols], partition='train')
mlca_runtime.save_model(pre, 'preprocessor')
