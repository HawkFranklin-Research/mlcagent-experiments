import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = protocol['targets']
excluded = set(protocol.get('excluded_columns', []) or [])
group_col = protocol.get('group_column')
target = targets[0]

candidate = [c for c in train.columns if c not in set(targets) | excluded and c != group_col]

def is_key_like(s):
    n = len(s)
    if s.dtype == object:
        return s.nunique(dropna=False) == n and n > 1
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n and n > 1
    return False

drop_keys = [c for c in candidate if is_key_like(train[c])]
features = [c for c in candidate if c not in drop_keys]

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = sum(1 for c in features if train[c].dtype == object
              and (set(train[c].dropna().astype(str).unique()) & ph_tokens))
print(f"Placeholder tokens found in {ph_cols} object columns")

y_tr = train[target]
y_is_numeric = pd.api.types.is_numeric_dtype(y_tr)
derived = []
for c in features:
    col = train[c]
    if col.dtype == y_tr.dtype and col.equals(y_tr):
        derived.append(c)
features = [c for c in features if c not in derived]
if derived:
    print(f"Excluded derived features: {derived[:5]} (n={len(derived)})")

if y_is_numeric:
    yv = pd.to_numeric(y_tr, errors='coerce').to_numpy(dtype=float)
    high = []
    for c in [c for c in features if pd.api.types.is_numeric_dtype(train[c])]:
        v = train[c].to_numpy(dtype=float)
        m = ~np.isnan(v) & ~np.isnan(yv)
        if m.sum() > 1:
            if abs(np.corrcoef(v[m], yv[m])[0, 1]) >= 0.98:
                high.append(c)
    if high:
        print(f"High-correlation features kept: {high[:5]} (n={len(high)})")

num_cols, cat_cols, date_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.dtype == object:
        if s.nunique(dropna=True) > max(1, int(0.05 * len(s))):
            continue
        cat_cols.append(c)
    else:
        date_cols.append(c)

X_train = train[num_cols + cat_cols].copy()
X_val = val[num_cols + cat_cols].copy()

for c in date_cols:
    dt_tr = pd.to_datetime(train[c], errors='coerce')
    dt_va = pd.to_datetime(val[c], errors='coerce')
    for part in ('month', 'dow', 'hour'):
        col = f"{c}_{part}"
        X_train[col] = getattr(dt_tr.dt, part if part != 'dow' else 'dayofweek').astype(float)
        X_val[col] = getattr(dt_va.dt, part if part != 'dow' else 'dayofweek').astype(float)
        num_cols.append(col)

for c in cat_cols:
    X_train[c] = X_train[c].astype(str).str.strip().replace(list(ph_tokens - {''}), np.nan)
    X_val[c] = X_val[c].astype(str).str.strip().replace(list(ph_tokens - {''}), np.nan)

n_out = len(num_cols) + int(sum(X_train[c].nunique(dropna=True) for c in cat_cols))
num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ct = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
if n_out <= 300:
    ct.set_output(transform='pandas')

mlca_runtime.fit_record(ct, X_train, 'train')

try:
    Xt = ct.transform(X_val)
    ncols = Xt.shape[1] if hasattr(Xt, 'shape') else -1
    print(f"Fitted preprocessor: {len(num_cols)} num + {len(cat_cols)} cat -> {ncols} cols; val rows={len(X_val)}")
except Exception as e:
    print(f"Transform check: {type(e).__name__}: {e}")

mlca_runtime.save_model(ct, 'preprocessor.pkl')
