import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise ValueError(f"missing key {k}; available top-level keys: {sorted(proto.keys())}")
targets = list(proto['targets'])
excluded = list(proto['excluded_columns'])
group = proto.get('group_column')
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

blocked = set(targets) | set(excluded) | ({group} if group else set())
cand = [c for c in train.columns if c not in blocked]


def keylike(s):
    if s.isna().any():
        return False
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique() == len(s)
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique() == len(s)
    return False


keys = [c for c in cand if keylike(train[c])]
feats = [c for c in cand if c not in keys]

target = targets[0]
derived = [c for c in feats if train[c].equals(train[target])]
for c in derived:
    print(f"excluded {c}: exact derivation of target {target}")
feats = [c for c in feats if c not in derived]

n = len(train)
num_cols, cat_cols, date_cols, drop_cols = [], [], [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) > 0.05 * n:
            drop_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        num_cols.append(c)

for d in date_cols:
    dt = pd.to_datetime(train[d], errors='coerce')
    dv = pd.to_datetime(val[d], errors='coerce')
    for part in ('year', 'month', 'dayofweek'):
        name = f"{d}_{part}"
        train[name] = getattr(dt.dt, part)
        val[name] = getattr(dv.dt, part)
        num_cols.append(name)

tokens = ['?', '-', 'NA', 'N/A']
ph_cols = [c for c in cat_cols
           if train[c].astype('string').str.strip().isin(tokens + ['']).any()]
print(f"feats={len(feats)} num={len(num_cols)} cat={len(cat_cols)} "
      f"dropped={len(drop_cols)} keys={len(keys)} placeholder_cols={len(ph_cols)}")

width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) + 1 for c in cat_cols)
parts = [('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                           ('sc', StandardScaler())]), num_cols),
         ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                           ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)]
parts = [p for p in parts if p[2]]
pre = ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)
if width <= 300:
    pre.set_output(transform='pandas')

cols = num_cols + cat_cols
Xtr = train[cols]
Xva = val.reindex(columns=cols)
fitted = mlca_runtime.fit_record(pre, Xtr, 'train', y=train[target])
Xt = fitted.transform(Xva)
names = fitted.get_feature_names_out() if width <= 300 else None
print(f"train {Xtr.shape} val {Xva.shape} -> encoded {Xt.shape}")
mlca_runtime.save_model(fitted, 'preprocessor.pkl')
