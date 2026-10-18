import mlca_runtime, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from pandas.api import types as pdt

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
avail = list(proto.keys())

def need(k):
    if k not in proto:
        raise KeyError(f"missing {k}; available={avail}")
    return proto[k]

targets = need('targets'); excluded = list(need('excluded_columns'))
group = need('group_column'); task = need('task')

hits = {}
for c in list(train.columns):
    s = train[c]
    if s.dtype == object or pdt.is_string_dtype(s):
        t = s.astype('string').str.strip()
        m = t.isin(['?', '-', 'NA', 'N/A', ''])
        if m.any():
            hits[c] = int(m.sum())
            t = t.mask(t.isin(['?', '-', 'NA', 'N/A', '']), pd.NA)
        train[c] = t.astype(object)
print(f"placeholders cols={len(hits)} count={sum(hits.values())}" if hits else "placeholders none")

for c in list(train.columns):
    s = train[c]; src = None
    if pdt.is_datetime64_any_dtype(s):
        src = s
    elif s.dtype == object:
        nn = s.dropna()
        if len(nn) > 0:
            try:
                if nn.astype(str).str.contains(r'[-:T ]').mean() > 0.9:
                    p = pd.to_datetime(nn, errors='coerce')
                    if p.notna().mean() > 0.9:
                        src = pd.to_datetime(s, errors='coerce')
            except Exception:
                pass
    if src is not None:
        for part, fn in [('year', lambda v: v.dt.year), ('month', lambda v: v.dt.month),
                         ('day', lambda v: v.dt.day), ('dow', lambda v: v.dt.dayofweek),
                         ('hour', lambda v: v.dt.hour)]:
            train[f'{c}_{part}'] = fn(src).astype(float)
        train = train.drop(columns=[c])

def is_key(s):
    if s.isna().any() or s.nunique() != len(s): return False
    if s.dtype == object or pdt.is_string_dtype(s): return True
    if pdt.is_integer_dtype(s):
        v = np.sort(s.values)
        if len(v) < 2: return False
        d = np.diff(v)
        return (d == d[0]).all()
    return False

banned = set(targets) | set(excluded) | {group}
feat = [c for c in train.columns if c not in banned and not is_key(train[c])]
ndrop_key = sum(1 for c in train.columns if c not in banned and is_key(train[c]))

n = len(train); num, cat, drop = [], [], []
for c in feat:
    s = train[c]
    if pdt.is_numeric_dtype(s):
        num.append(c)
    elif s.dtype == object or pdt.is_string_dtype(s):
        if s.nunique(dropna=True) > 0.05 * n: drop.append(c)
        else: cat.append(c)
    else:
        drop.append(c)

def np_pipe(c):
    s = train[c].dropna()
    steps = [('imp', SimpleImputer(strategy='median'))]
    if len(s) > 1 and (s >= 0).all() and abs(float(s.skew())) > 2:
        steps.append(('log', FunctionTransformer(np.log1p)))
    steps.append(('sc', StandardScaler()))
    return Pipeline(steps)

def cat_pipe():
    return Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

width = len(num) + sum(int(train[c].nunique(dropna=True)) for c in cat)
use_pd = width <= 300
trs = [(f'n_{c}', np_pipe(c), [c]) for c in num] + [(f'c_{c}', cat_pipe(), [c]) for c in cat]
ct = ColumnTransformer(trs, remainder='drop', verbose_feature_names_out=False)
if use_pd:
    ct.set_output(transform='pandas')

mlca_runtime.fit_record(ct, train[feat], partition='train')
mlca_runtime.save_model(ct, 'preprocessor.pkl')
print(f"num={len(num)} cat={len(cat)} drop={len(drop)} keys={ndrop_key} width~{width} pandas_out={use_pd}")
