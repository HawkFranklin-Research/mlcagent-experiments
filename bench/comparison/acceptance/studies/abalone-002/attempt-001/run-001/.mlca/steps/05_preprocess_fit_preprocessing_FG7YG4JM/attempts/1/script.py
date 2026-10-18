import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from itertools import combinations

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group_col = proto.get('group_column')
train = pd.read_parquet(inputs['train'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)


def is_id(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        return False
    if s.isna().any() or s.nunique() != len(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        v = np.sort(s.values.astype(float))
        return bool(np.allclose(np.diff(v), 1))
    return True


feats = [c for c in train.columns if c not in drop and not is_id(train[c])]
tgt = targets[0]
y = train[tgt]

derived = []
num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
for c in num_feats:
    if np.allclose(train[c].astype(float), y.astype(float), equal_nan=True):
        derived.append((c, 'equals target'))
if pd.api.types.is_numeric_dtype(y):
    nn = [c for c in num_feats if (train[c] >= 0).all() and c not in [d[0] for d in derived]]
    for r in [2, 3]:
        for combo in combinations(nn, r):
            s = sum(train[c].astype(float) for c in combo)
            if np.allclose(s, y.astype(float), equal_nan=True):
                for c in combo:
                    if not any(d[0] == c for d in derived):
                        derived.append((c, 'sum equals target'))
for c, r in derived:
    print(f"excluded feature {c}: {r}")
    feats.remove(c)
num_feats = [f for f in num_feats if f in feats]

if pd.api.types.is_numeric_dtype(y):
    warns = []
    for c in num_feats:
        if train[c].std() > 0:
            co = np.corrcoef(train[c].astype(float), y.astype(float))[0, 1]
            if abs(co) >= 0.98:
                warns.append(c)
    if warns:
        print("correlation>=0.98 with target: " + ", ".join(warns))

n = len(train)
num_cols, cat_cols = [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nuv = s.nunique()
        if nuv <= max(20, int(0.05 * n)):
            cat_cols.append(c)
        else:
            print(f"dropping high-cardinality {c} ({nuv} unique)")

ph_cols = 0
for c in cat_cols:
    s = train[c].astype(str)
    stripped = s.str.strip()
    m = (stripped != s) | stripped.isin(['?', '-', 'NA', 'N/A', ''])
    if m.any():
        ph_cols += 1
if ph_cols:
    print(f"placeholder tokens found in {ph_cols} columns")

train_clean = train[feats].copy()
for c in cat_cols:
    s = train_clean[c].astype(str).str.strip()
    train_clean[c] = s.where(~s.isin(['?', '-', 'NA', 'N/A', '']), np.nan)

est = len(num_cols) + sum(int(train[c].nunique()) for c in cat_cols)
use_pandas = est <= 300

trans = []
if num_cols:
    trans.append(('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
    ]), num_cols))
if cat_cols:
    trans.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

pre = ColumnTransformer(transformers=trans, remainder='drop')
if use_pandas:
    pre.set_output(transform='pandas')

mlca_runtime.fit_record(pre, train_clean)
mlca_runtime.save_model(pre, 'preprocessor.pkl')

Xt = pre.transform(train_clean)
print(f"features={len(feats)} num={len(num_cols)} cat={len(cat_cols)} out_dim={Xt.shape[1]}")
