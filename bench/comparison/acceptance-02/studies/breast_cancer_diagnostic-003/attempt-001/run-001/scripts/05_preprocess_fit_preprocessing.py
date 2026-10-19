import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
for k in ('targets', 'excluded_columns'):
    if k not in protocol:
        raise KeyError(f"missing {k}; available top-level keys: {list(protocol.keys())}")
train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])

targets = list(protocol['targets'])
target_col = targets[0]
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

def is_keylike(s, n):
    if pd.api.types.is_integer_dtype(s) and s.nunique() == n:
        return True
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == n:
        return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded
             and (group is None or c != group)
             and not is_keylike(train[c], len(train))]

ph_cols = 0
for c in feat_cols:
    if train[c].dtype == object:
        vals = train[c].dropna().astype(str)
        if len(vals) and (vals.isin(['?', '-', 'NA', 'N/A']).any()
                          or (vals != vals.str.strip()).any()):
            ph_cols += 1
print(f"placeholder_cols={ph_cols}")

t = train[target_col]
num_all = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]
derived = set()
if pd.api.types.is_numeric_dtype(t):
    tv = t.astype(float).values
    for c in num_all:
        if np.allclose(train[c].astype(float).values, tv, equal_nan=True):
            derived.add(c)
    nn = [c for c in num_all if (train[c].dropna() >= 0).all()]
    for k in (2, 3):
        for combo in combinations(nn, k):
            s = sum(train[c].fillna(0) for c in combo).values
            if np.allclose(s, tv):
                derived.update(combo)
else:
    tcode = pd.factorize(t)[0].astype(float)
    for c in num_all:
        if np.allclose(train[c].fillna(-99).astype(float).values, tcode):
            derived.add(c)
for c in sorted(derived):
    print(f"excluded {c}: derived from target")
feat_cols = [c for c in feat_cols if c not in derived]

if pd.api.types.is_numeric_dtype(t):
    tv = t.astype(float)
    warn = [c for c in num_all if c in feat_cols
            and abs(train[c].astype(float).corr(tv)) >= 0.98]
    if warn:
        print(f"high_corr_warning={warn}")

n = len(train)
num_cols, cat_cols, hc_text, dt_cols = [], [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nu = s.nunique(dropna=True)
        (hc_text if nu > max(2, int(0.05 * n)) else cat_cols).append(c)

for c in dt_cols:
    for part, fn in [('hour', lambda x: x.dt.hour),
                     ('dow', lambda x: x.dt.dayofweek),
                     ('month', lambda x: x.dt.month)]:
        nc = f"{c}__{part}"
        train[nc] = fn(pd.to_datetime(train[c]))
        val[nc] = fn(pd.to_datetime(val[c]))
        num_cols.append(nc)
for c in hc_text:
    print(f"dropped_hc_text={c}")
feat_cols = [c for c in feat_cols if c not in dt_cols and c not in hc_text]

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
use_pandas = width <= 300

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore',
                                          sparse_output=not use_pandas))])
trans = [('num', num_pipe, num_cols)]
if cat_cols:
    trans.append(('cat', cat_pipe, cat_cols))
pre = ColumnTransformer(trans, remainder='drop')
if use_pandas:
    pre.set_output(transform='pandas')

mlca_runtime.fit_record(pre, train[feat_cols], 'train', y=train[target_col])
Xv = pre.transform(val[feat_cols])
print(f"train_rows={len(train)} features={len(feat_cols)} num={len(num_cols)} cat={len(cat_cols)} val_out={Xv.shape}")

mlca_runtime.save_model(pre, 'preprocessor.pkl')
