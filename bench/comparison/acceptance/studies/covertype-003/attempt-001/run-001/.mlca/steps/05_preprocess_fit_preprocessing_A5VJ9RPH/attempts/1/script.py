import mlca_runtime
import json, itertools
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
avail = list(protocol.keys())
def get(k):
    if k not in protocol:
        raise KeyError(f"Missing '{k}'. Available: {avail}")
    return protocol[k]
targets = get("targets")
excluded = get("excluded_columns") or []
group = get("group_column")
task = get("task")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
n_train = len(train)

drop_set = set(targets) | set(excluded) | ({group} if group else set())
cands = [c for c in train.columns if c not in drop_set]

def key_like(c):
    s = train[c]
    if s.dtype == object:
        return s.nunique() == n_train
    if s.dtype.kind in 'iu':
        return s.nunique() == n_train
    return False
cands = [c for c in cands if not key_like(c)]

y = train[targets[0]].values
feat_exc = set()
for c in cands:
    if np.array_equal(train[c].values, y):
        feat_exc.add(c)
        print(f"Excluded {c}: equals target exactly")
nonneg = [c for c in cands if train[c].dtype.kind in 'iu' and train[c].min() >= 0]
if nonneg:
    ymax, ymin = y.max(), y.min()
    maxes = sorted([train[c].max() for c in nonneg], reverse=True)
    mins = sorted([train[c].min() for c in nonneg])
    for k in (2, 3):
        if len(nonneg) < k: continue
        if sum(maxes[:k]) < ymax or sum(mins[:k]) > ymin:
            continue
        for combo in itertools.combinations(nonneg, k):
            s = sum(train[c].values for c in combo)
            if np.array_equal(s, y):
                feat_exc.update(combo)
                print(f"Excluded {combo}: sum equals target")
                break
feature_cols = [c for c in cands if c not in feat_exc]

num_cols, cat_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'category':
        nun = s.nunique()
        if nun <= max(2, int(0.05 * n_train)):
            cat_cols.append(c)
        else:
            high_card.append(c)
    elif s.dtype.kind in 'iu':
        uniq = set(s.dropna().unique())
        if uniq <= {0, 1}:
            num_cols.append(c)
        else:
            nun = s.nunique()
            if nun <= 20 and nun <= 0.05 * n_train:
                cat_cols.append(c)
            else:
                num_cols.append(c)
    else:
        num_cols.append(c)
if high_card:
    print(f"High-cardinality dropped: {len(high_card)}")

ph_cols = 0
for c in train.columns:
    if train[c].dtype == object:
        s = train[c].astype(str)
        for tok in ['?', '-', 'NA', 'N/A']:
            if (s == tok).any() or s.str.strip().eq(tok).any():
                ph_cols += 1
                break
        else:
            if (s.str.strip() == '').any():
                ph_cols += 1
if ph_cols:
    print(f"Placeholder tokens in {ph_cols} text columns")

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
], remainder='drop')
pre.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[targets[0]]
mlca_runtime.fit_record(pre, X_train, 'train', y=y_train)
Xt = pre.transform(val[feature_cols])
print(f"Features {len(feature_cols)} (num {len(num_cols)}, cat {len(cat_cols)}); encoded val {Xt.shape[1]} cols")
mlca_runtime.save_model(pre, "preprocessor.pkl")
