import mlca_runtime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(Path(inp['_protocol']).read_text())

try:
    targets = protocol['targets']
    ex_raw = protocol['excluded_columns']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}")

excluded = set(ex_raw if not isinstance(ex_raw, dict) else ex_raw.get('examples', []))
group = protocol.get('group_column')
if len(targets) != 1:
    raise ValueError("This preprocess step expects exactly one target.")
y_col = targets[0]
y = train[y_col]

# Placeholder cleaning for object/string columns (deterministic, no fitted state)
tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for c in train.columns:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        s = train[c].where(train[c].isna(), s)
        ph = s.isin(tokens).sum()
        if ph:
            ph_cols += 1
            train[c] = s.replace(list(tokens), np.nan)
            if c in val.columns:
                v = val[c].astype(str).str.strip()
                val[c] = val[c].where(val[c].isna(), v).replace(list(tokens), np.nan)

# Initial feature columns: exclude targets, excluded columns, group column
cand = [c for c in train.columns if c not in targets and c not in excluded and c != group]

# Key-like column drop (all-distinct text codes or integer sequences)
drop_keys = []
for c in list(cand):
    s = train[c].dropna()
    if len(s) == 0:
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            drop_keys.append(c)
    elif pd.api.types.is_integer_dtype(s) and not set(s.unique()).issubset({0, 1}):
        if s.nunique() == len(s):
            drop_keys.append(c)
cand = [c for c in cand if c not in drop_keys]

# Screen for confirmed target derivations on training rows
excluded_deriv = []
deriv_names = set()
if pd.api.types.is_numeric_dtype(y):
    num_c = [c for c in cand if pd.api.types.is_numeric_dtype(train[c])]
    for c in num_c:
        m = train[c].notna() & y.notna()
        if m.sum() and (train.loc[m, c] == y[m]).all():
            excluded_deriv.append(([c], 'exact target'))
    corrs = train[num_c].corrwith(y).abs().sort_values(ascending=False)
    top = [c for c in corrs.index[:30] if c not in deriv_names]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            if (train[a] >= 0).all() and (train[b] >= 0).all():
                s = train[a] + train[b]
                m = s.notna() & y.notna()
                if m.sum() and (s[m] == y[m]).all():
                    excluded_deriv.append(([a, b], f'sum {a}+{b}'))
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            for k in range(j + 1, len(top)):
                a, b, c = top[i], top[j], top[k]
                if (train[a] >= 0).all() and (train[b] >= 0).all() and (train[c] >= 0).all():
                    s = train[a] + train[b] + train[c]
                    m = s.notna() & y.notna()
                    if m.sum() and (s[m] == y[m]).all():
                        excluded_deriv.append(([a, b, c], 'sum of three'))
    for names, _ in excluded_deriv:
        deriv_names.update(names)
    if excluded_deriv:
        print("Excluded target derivations:", "; ".join(f"{n} ({r})" for n, r in excluded_deriv[:5]))
    warn = [c for c in num_c if abs(corrs.get(c, 0)) >= 0.98 and c not in deriv_names]
    if warn:
        print("Correlation >=0.98 (kept):", ", ".join(warn[:10]))

cand = [c for c in cand if c not in deriv_names]

# Date handling: derive numeric parts, drop raw datetime column
for c in list(cand):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for part in ['year', 'month', 'day', 'dayofweek', 'hour']:
            nc = f"{c}_{part}"
            train[nc] = getattr(train[c].dt, part)
            val[nc] = getattr(val[c].dt, part)
            cand.append(nc)
        cand.remove(c)

# Type split: numeric vs low-cardinality categorical
num_cols, cat_cols = [], []
for c in cand:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(5, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            print(f"Dropping high-cardinality text/code column {c} ({nun} unique)")

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
use_sparse = width > 300

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler()),
])
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=use_sparse)),
])
transformers = []
if num_cols:
    transformers.append(('num', num_pipe, num_cols))
if cat_cols:
    transformers.append(('cat', cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder='drop')
if not use_sparse:
    pre.set_output(transform='pandas')
    num_pipe.set_output(transform='pandas')
    cat_pipe.set_output(transform='pandas')

X_train = train[num_cols + cat_cols].copy()
X_val = val[num_cols + cat_cols].copy()
fitted = mlca_runtime.fit_record(pre, X_train, 'train', y=y)
Xv = fitted.transform(X_val)
mlca_runtime.save_model(fitted, "preprocessor.pkl")

print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical; width={width}, sparse={use_sparse}")
print(f"Train rows={len(X_train)}, val rows={len(X_val)}, val transform shape={Xv.shape}")
print(f"Placeholder columns: {ph_cols}; key-like dropped: {len(drop_keys)}")
