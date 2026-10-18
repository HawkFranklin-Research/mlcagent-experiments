import mlca_runtime, json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])
protocol = json.loads(paths['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns', []) or [])
group = protocol.get('group_column')
y_tr = train[targets[0]]
skip = set(targets) | set(excluded)
if group: skip.add(group)

ph = {'?', '-', 'NA', 'N/A'}
n_ph = 0
for c in train.columns:
    if c in skip: continue
    if pd.api.types.is_string_dtype(train[c]) or train[c].dtype == object:
        s_tr = train[c].astype('string').str.strip()
        s_va = val[c].astype('string').str.strip()
        if s_tr.isin(list(ph) + ['']).sum() > 0: n_ph += 1
        train[c] = s_tr.mask(s_tr.isin(list(ph) + ['']).fillna(False))
        val[c] = s_va.mask(s_va.isin(list(ph) + ['']).fillna(False))
print(f"placeholder cols: {n_ph}")

cand = [c for c in train.columns if c not in skip]
y_vals = y_tr.astype(float).values

derived = set()
for c in cand:
    col = train[c]
    if not pd.api.types.is_numeric_dtype(col) or col.isna().any(): continue
    if np.array_equal(col.astype(float).values, y_vals): derived.add(c)
nonneg = [c for c in cand if pd.api.types.is_numeric_dtype(train[c]) and train[c].dropna().min() >= 0]
for r in (2, 3):
    for combo in combinations(nonneg, r):
        s = train[list(combo)].sum(axis=1).values.astype(float)
        if np.array_equal(s, y_vals): derived.update(combo); break
if derived: print(f"excluded derivation: {sorted(derived)}")

feature_cols = [c for c in cand if c not in derived]

warn = []
if pd.api.types.is_numeric_dtype(y_tr) and y_tr.nunique() == 2:
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            cc = train[c].corr(y_tr.astype(float))
            if pd.notna(cc) and abs(cc) >= 0.98: warn.append(c)
if warn: print(f"corr>=0.98 kept: {warn}")

key_like = [c for c in feature_cols
            if pd.api.types.is_integer_dtype(train[c]) and train[c].notna().all()
            and train[c].nunique() == len(train) and train[c].nunique() >= 20]
if key_like:
    print(f"drop key-like: {key_like}")
    feature_cols = [c for c in feature_cols if c not in key_like]

num_cols = [c for c in feature_cols
            if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
thr = 0.05 * len(train)
high = [c for c in cat_cols if train[c].nunique(dropna=True) > thr]
if high:
    print(f"drop high-card: {high}")
    cat_cols = [c for c in cat_cols if c not in high]
    feature_cols = [c for c in feature_cols if c not in high]

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ct = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
ct.set_output(transform='pandas')

ct = mlca_runtime.fit_record(ct, train[feature_cols], 'train', y=None)
Xt_val = ct.transform(val[feature_cols])
print(f"feature_cols={len(feature_cols)} transformed_val={Xt_val.shape}")
mlca_runtime.save_model(ct, 'preprocessor.pkl')
