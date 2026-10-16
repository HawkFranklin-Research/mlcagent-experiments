import mlca_runtime, json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

candidates = [c for c in train.columns if c not in drop]

feature_cols = []
for c in candidates:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s):
        if s.nunique(dropna=True) == len(s) and len(s) > 1:
            continue
    feature_cols.append(c)

n = len(train)
num_cols, cat_cols, text_cols, date_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(1, int(0.05 * n)):
            cat_cols.append(c)
        else:
            text_cols.append(c)

placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in cat_cols + text_cols:
    s = train[c].astype(str)
    if s.str.strip().isin(placeholders).any() or (s != s.str.strip()).any():
        ph_cols.append(c)
print(f"placeholder columns: {len(ph_cols)}")

log_cols, plain_num = [], []
for c in num_cols:
    s = train[c].dropna()
    if len(s) > 0 and (s >= 0).all() and s.skew() > 2:
        log_cols.append(c)
    else:
        plain_num.append(c)

transformers = []
if plain_num:
    transformers.append(('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), plain_num))
if log_cols:
    transformers.append(('log', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ]), log_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

ct = ColumnTransformer(transformers, remainder='drop')
ct.set_output(transform='pandas')

mlca_runtime.fit_record(ct, train[feature_cols], partition='train')
mlca_runtime.save_model(ct, 'preprocessor.pkl')

print(f"features={len(feature_cols)} num={len(plain_num)} log={len(log_cols)} cat={len(cat_cols)}")
