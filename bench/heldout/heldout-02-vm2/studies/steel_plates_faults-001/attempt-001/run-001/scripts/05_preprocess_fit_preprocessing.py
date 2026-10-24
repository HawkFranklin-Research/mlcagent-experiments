import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
prot = json.loads(inputs['_protocol'].read_text())

def _k(k):
    if k not in prot:
        raise KeyError(f"{k} missing; available: {list(prot.keys())}")
    return prot[k]

targets = _k('targets')
excluded = _k('excluded_columns')
group_col = _k('group_column')
_ = _k('task')

drop = set(targets) | set(excluded)
if group_col is not None and group_col in train.columns:
    drop.add(group_col)
cand = [c for c in train.columns if c not in drop]

key_like = []
for c in cand:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) == len(s.dropna()):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s.dropna()) and s.max() - s.min() + 1 == len(s.dropna()):
            key_like.append(c)
if key_like:
    cand = [c for c in cand if c not in key_like]

numeric_cand = [c for c in cand
                if pd.api.types.is_numeric_dtype(train[c])
                and not pd.api.types.is_bool_dtype(train[c])]
excluded_deriv = set()
exclusion_log = []
for t in targets:
    y = train[t]
    if not pd.api.types.is_numeric_dtype(y):
        continue
    for c in numeric_cand:
        if c in excluded_deriv:
            continue
        if (train[c].fillna(y) == y).all():
            excluded_deriv.add(c); exclusion_log.append(f"{c}=eq({t})")
    nn = [c for c in numeric_cand if c not in excluded_deriv and (train[c].dropna() >= 0).all()]
    for i in range(len(nn)):
        for j in range(i + 1, len(nn)):
            a, b = nn[i], nn[j]
            if ((train[a] + train[b]).fillna(y) == y).all():
                for c in (a, b):
                    if c not in excluded_deriv:
                        excluded_deriv.add(c); exclusion_log.append(f"{c}=sum2({t})")
    for i in range(len(nn)):
        for j in range(i + 1, len(nn)):
            for m in range(j + 1, len(nn)):
                a, b, c3 = nn[i], nn[j], nn[m]
                if ((train[a] + train[b] + train[c3]).fillna(y) == y).all():
                    for c in (a, b, c3):
                        if c not in excluded_deriv:
                            excluded_deriv.add(c); exclusion_log.append(f"{c}=sum3({t})")

final = [c for c in cand if c not in excluded_deriv]

num_cols, cat_cols, dt_cols, drop_high = [], [], [], []
for c in final:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > 0.05 * len(train):
            drop_high.append(c)
        else:
            cat_cols.append(c)
    else:
        drop_high.append(c)

placeholders = ['?', '-', 'NA', 'N/A', '']
ph_hits = 0
for c in cat_cols:
    ser = train[c]
    if isinstance(ser.dtype, pd.CategoricalDtype):
        ser = ser.astype('object')
    stripped = ser.astype('object').str.strip()
    hit = stripped.isin(placeholders).any()
    if hit:
        ph_hits += 1
    train[c] = stripped.where(~stripped.isin(placeholders), other=np.nan)

X_train = train[final].copy()
steps = []
if num_cols:
    steps.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                    ('sc', StandardScaler())]), num_cols))
if cat_cols:
    steps.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                    ('oh', OneHotEncoder(handle_unknown='ignore',
                                                         sparse_output=False))]), cat_cols))
pre = ColumnTransformer(steps, remainder='drop')
pre.set_output(transform='pandas')
mlca_runtime.fit_record(pre, X_train, 'train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')

print(f"Features used: {len(final)} (num={len(num_cols)}, cat={len(cat_cols)})")
if exclusion_log:
    print("Excluded derived: " + "; ".join(exclusion_log[:6]))
if dt_cols:
    print("Dropped datetime: " + ", ".join(dt_cols))
if drop_high:
    print("Dropped high-cardinality: " + ", ".join(drop_high))
if ph_hits:
    print(f"Placeholder tokens found in {ph_hits} categorical columns")
