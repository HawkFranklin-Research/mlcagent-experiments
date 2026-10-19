import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
protocol = json.loads(inputs['_protocol'].read_text())

avail = list(protocol.keys())
def need(k):
    if k not in protocol:
        raise KeyError(f"Missing key {k}; available: {avail}")
    return protocol[k]

targets = need('targets') or []
excluded = need('excluded_columns') or []
group = protocol.get('group_column')

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = [c for c in train.columns if c not in drop]
n = len(train)

key_like = [c for c in feature_cols
            if pd.api.types.is_integer_dtype(train[c])
            and train[c].notna().all() and train[c].nunique() == n]
for c in key_like:
    feature_cols.remove(c)
    print(f"dropped key-like {c}")

num_cols, cat_cols, bool_cols, date_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_bool_dtype(s):
        bool_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

log_cols, plain_num = [], []
for c in num_cols:
    v = train[c].dropna()
    if len(v) and (v >= 0).all() and v.skew() > 2:
        log_cols.append(c)
    else:
        plain_num.append(c)

PH = {'?', '-', 'NA', 'N/A', ''}
if cat_cols:
    found = {c: int(train[c].astype(str).str.strip().isin(PH).sum()) for c in cat_cols}
    found = {k: v for k, v in found.items() if v}
    if found:
        print(f"placeholders in {len(found)} cat cols")

transformers = []
if plain_num:
    transformers.append(('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), plain_num))
if log_cols:
    transformers.append(('lognum', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ]), log_cols))
if bool_cols:
    transformers.append(('bool', 'passthrough', bool_cols))
if date_cols:
    def date_parts(X):
        X = X.copy()
        for c in X.columns:
            d = pd.to_datetime(X[c])
            X[c + '_year'] = d.dt.year
            X[c + '_month'] = d.dt.month
            X[c + '_dow'] = d.dt.dayofweek
            X[c + '_day'] = d.dt.day
        keep = [c for c in X.columns if not pd.api.types.is_datetime64_any_dtype(X[c])]
        return X[keep]
    transformers.append(('date', Pipeline([
        ('parts', FunctionTransformer(date_parts)),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), date_cols))
if cat_cols:
    def clean_text(X):
        X = X.copy()
        for c in X.columns:
            s = X[c].astype(str).str.strip()
            X[c] = s.mask(s.isin(PH))
        return X
    transformers.append(('cat', Pipeline([
        ('clean', FunctionTransformer(clean_text)),
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

if not transformers:
    raise ValueError("No features selected for preprocessing")

pre = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
pre.set_output(transform='pandas')

mlca_runtime.fit_record(pre, train[feature_cols], partition='train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')
print(f"preprocessor fit on {len(train)} rows, {len(feature_cols)} features")
