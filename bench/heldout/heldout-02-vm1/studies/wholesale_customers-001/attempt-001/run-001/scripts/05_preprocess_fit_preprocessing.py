import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
P = json.loads(paths['_protocol'].read_text())
for k in ('targets', 'excluded_columns'):
    if k not in P:
        raise ValueError(f"missing key {k}; available top-level: {list(P.keys())}")
targets = list(P['targets'])
excluded = list(P['excluded_columns'])
group = P.get('group_column')

train = pd.read_parquet(paths['train'])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith('str')) and \
       s.dropna().is_unique and s.nunique(dropna=True) > 20:
        continue
    feat_cols.append(c)

ph_tokens = ['?', '-', 'NA', 'N/A']
ph_count = 0
for c in feat_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith('str'):
        v = s.dropna().astype(str)
        if v.isin(ph_tokens).any() or (v.str.strip() == '').any() or (v != v.str.strip()).any():
            ph_count += 1
print(f"placeholders: {ph_count} text cols")

num_cols, cat_cols, dt_cols, high_cols = [], [], [], []
n = len(train)
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) / max(n, 1) > 0.05:
            high_cols.append(c)
        else:
            cat_cols.append(c)

for c in dt_cols:
    d = pd.to_datetime(train[c], errors='coerce')
    for suffix, part in (('dow', d.dt.dayofweek), ('month', d.dt.month), ('hour', d.dt.hour)):
        name = f"{c}__{suffix}"
        train[name] = part
        num_cols.append(name)

if high_cols:
    print(f"drop high-cardinality: {high_cols}")
print(f"features num={len(num_cols)} cat={len(cat_cols)} dt={len(dt_cols)} high={len(high_cols)}")

def num_pipe(s):
    steps = []
    sv = s.astype(float).dropna()
    if len(sv) and (sv >= 0).all() and float(sv.skew()) > 1.0:
        steps.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')))
    steps.append(('imp', SimpleImputer(strategy='median')))
    steps.append(('sc', StandardScaler()))
    return Pipeline(steps)

transformers = [(f"num_{c}", num_pipe(train[c]), [c]) for c in num_cols]
if cat_cols:
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    transformers.append(('cat', cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder='drop')
pre.set_output(transform='pandas')
input_cols = num_cols + cat_cols
mlca_runtime.fit_record(pre, train[input_cols], partition='train')
mlca_runtime.save_model(pre, "preprocessor.pkl")
