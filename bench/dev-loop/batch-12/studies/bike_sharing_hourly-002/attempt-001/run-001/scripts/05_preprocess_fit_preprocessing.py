import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(Path(inputs['_protocol']).read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise KeyError(f"Missing '{k}' in _protocol. Available: {list(proto.keys())}")
if 'primary_metric' not in proto['task']:
    raise KeyError(f"Missing 'primary_metric' in _protocol['task']. Available: {list(proto['task'].keys())}")

targets = proto['targets'] if isinstance(proto['targets'], list) else [proto['targets']]
excluded = list(proto['excluded_columns'])
group_col = proto.get('group_column')
train = pd.read_parquet(inputs['train'])
n = len(train)
drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in drop]

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

def classify(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return 'date'
    if pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        if len(u) <= 2 and set(u).issubset({0, 1, 0.0, 1.0, False, True}):
            return 'num'
        if pd.api.types.is_integer_dtype(s) and len(u) <= 20:
            return 'cat'
        return 'num'
    smp = s.dropna().head(50)
    if len(smp):
        try:
            pd.to_datetime(smp, errors='raise')
            return 'date'
        except Exception:
            pass
    if s.nunique(dropna=True) > max(1, 0.05 * n):
        return 'drop'
    return 'cat'

for c in list(feature_cols):
    s = train[c]
    if s.nunique(dropna=True) == n and (is_text(s) or
        (pd.api.types.is_integer_dtype(s) and np.ptp(np.sort(s.dropna().unique())) == n - 1)):
        print(f"Dropped key-like: {c}")
        drop.add(c); feature_cols.remove(c)

tgt = train[targets[0]].to_numpy()
derived = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]) and np.array_equal(train[c].to_numpy(), tgt, equal_nan=True):
        derived.append((c, 'exact-equals target'))
nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and not (train[c].fillna(0) < 0).any()]
for size in (2, 3):
    for combo in combinations(nonneg, size):
        s = sum((train[cc].fillna(0).to_numpy() for cc in combo), np.zeros(n))
        if np.allclose(s, tgt):
            for cc in combo:
                if cc not in [d[0] for d in derived]:
                    derived.append((cc, 'sum-derived'))
for c, r in derived:
    print(f"Dropped derived: {c} ({r})")
    drop.add(c); feature_cols.remove(c)

if pd.api.types.is_numeric_dtype(train[targets[0]]):
    warns = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            try:
                r = float(np.corrcoef(train[c].fillna(train[c].median()).to_numpy(dtype=float), tgt.astype(float))[0, 1])
                if abs(r) >= 0.98:
                    warns.append((c, r))
            except Exception:
                pass
    if warns:
        print("Corr>=0.98 kept: " + ", ".join(f"{c}={r:.3f}" for c, r in warns))

tokens = {'?', '-', 'NA', 'N/A', ''}
ph = [c for c in feature_cols if is_text(train[c]) and train[c].dropna().astype(str).str.strip().isin(tokens).any()]
print(f"Placeholders in {len(ph)} columns")

buckets = {'num': [], 'cat': [], 'date': [], 'drop': []}
for c in feature_cols:
    buckets[classify(train[c])].append(c)
for c in buckets['drop']:
    print(f"Dropped high-card: {c}")
    drop.add(c); feature_cols.remove(c)

trans = []
if buckets['num']:
    trans.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), buckets['num']))
if buckets['cat']:
    trans.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                   ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), buckets['cat']))
if buckets['date']:
    trans.append(('dt', Pipeline([('oe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                                  ('sc', StandardScaler())]), buckets['date']))

ct = ColumnTransformer(trans, remainder='drop')
for _, t, _ in trans:
    if hasattr(t, 'set_output'):
        t.set_output(transform='pandas')
ct.set_output(transform='pandas')

X = train[feature_cols]
y = train[targets[0]]
pre = mlca_runtime.fit_record(ct, X, 'train', y=y)
width = pre.transform(X.head(5)).shape[1]
print(f"Features: {len(feature_cols)} -> {width} encoded cols")
mlca_runtime.save_model(pre, 'preprocessor.pkl')
