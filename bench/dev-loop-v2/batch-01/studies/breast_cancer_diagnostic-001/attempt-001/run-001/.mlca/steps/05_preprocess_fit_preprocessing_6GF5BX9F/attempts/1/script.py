import mlca_runtime, json
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise ValueError(f"missing {k}; available: {list(proto.keys())}")
targets = proto['targets']; excluded = proto['excluded_columns']
group = proto.get('group_column')
train = pd.read_parquet(inp['train']); val = pd.read_parquet(inp['val'])

blocked = set(targets) | set(excluded) | ({group} if group else set())
feats = [c for c in train.columns if c not in blocked]

def is_keylike(s):
    if pd.api.types.is_numeric_dtype(s) and s.dtype.kind in 'iu':
        return s.notna().all() and s.nunique() == len(s)
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().all() and s.nunique() == len(s)
    return False

keys = [c for c in feats if is_keylike(train[c])]
feats = [c for c in feats if c not in keys]

tgt = train[targets[0]] if targets[0] in train else None
derived = [c for c in feats if tgt is not None and train[c].equals(tgt)]
for c in derived:
    print(f"excluded {c}: exact derivation of target")
feats = [c for c in feats if c not in derived]

n = len(train)
num_cols, cat_cols, date_cols, drop_cols = [], [], [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) > 0.05 * n:
            drop_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        num_cols.append(c)

for d in date_cols:
    dt = pd.to_datetime(train[d], errors='coerce'); dv = pd.to_datetime(val[d], errors='coerce')
    for part, fn_t, fn_v in [('year', dt.dt.year, dv.dt.year),
                             ('month', dt.dt.month, dv.dt.month),
                             ('dayofweek', dt.dt.dayofweek, dv.dt.dayofweek)]:
        name = f"{d}_{part}"
        train[name] = fn_t; val[name] = fn_v
        num_cols.append(name)

ph_cols = []
for c in cat_cols:
    s = train[c].astype('string').str.strip()
    if s.isin(['?', '-', 'NA', 'N/A']).any() or s.eq('').any():
        ph_cols.append(c)
print(f"feats={len(feats)} num={len(num_cols)} cat={len(cat_cols)} drops={len(drop_cols)} key={len(keys)} ph={len(ph_cols)}")

pre = ColumnTransformer(
    [('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                       ('sc', StandardScaler())]), num_cols),
     ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                       ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)],
    remainder='drop')
pre.set_output(transform='pandas')

Xtr = train[num_cols + cat_cols].copy()
Xva = val.reindex(columns=Xtr.columns).copy()
y = train[targets[0]] if targets[0] in train else None
fitted = mlca_runtime.fit_record(pre, Xtr, 'train', y=y)
Xt = fitted.transform(Xva)
print(f"train {Xtr.shape} val {Xva.shape} -> transformed {Xt.shape}")
mlca_runtime.save_model(fitted, 'preprocessor')
