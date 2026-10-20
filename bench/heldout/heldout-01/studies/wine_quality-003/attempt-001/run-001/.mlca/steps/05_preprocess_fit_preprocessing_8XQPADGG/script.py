import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
for t in targets:
    if t not in train.columns:
        raise KeyError(f"Target {t} not in train columns: {list(train.columns)}")
y = train[targets[0]].copy()
n = len(train)

def is_key_like(s, name):
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == n:
            if 'id' in name.lower():
                return True
            vals = s.dropna().sort_values().values
            if len(vals) == n and vals[0] in (0, 1) and np.all(np.diff(vals) == 1):
                return True
    return False

candidate = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in candidate if not is_key_like(train[c], c)]
if len(feature_cols) < len(candidate):
    print(f"Dropped key-like columns: {sorted(set(candidate) - set(feature_cols))}")

tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(tokens).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholder tokens in {len(ph_cols)} columns: {ph_cols}")
    for c in ph_cols:
        for df in (train, val):
            df[c] = df[c].astype(str).str.strip().replace(list(tokens), np.nan)

numeric_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        nu = s.nunique(dropna=True)
        if nu > max(5, 0.05 * n):
            drop_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        drop_cols.append(c)
if drop_cols:
    print(f"Dropped non-numeric/high-cardinality columns: {drop_cols}")

if pd.api.types.is_numeric_dtype(y):
    y_arr = y.values
    nonneg = [c for c in numeric_cols if (train[c].dropna() >= 0).all()]
    found = True
    while found:
        found = False
        for r in (1, 2, 3):
            for combo in combinations(nonneg, r):
                if any(c not in numeric_cols for c in combo):
                    continue
                if np.allclose(train[list(combo)].sum(axis=1).values, y_arr, equal_nan=False):
                    for c in combo:
                        if c in numeric_cols:
                            numeric_cols.remove(c)
                            if c in feature_cols:
                                feature_cols.remove(c)
                            print(f"Excluded derived feature {c}: sum of {combo} equals target")
                    found = True
                    break
            if found:
                break

n_out_est = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
sparse = n_out_est > 300
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse))])
preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')
if not sparse:
    preprocessor.set_output(transform='pandas')

X_train = train[numeric_cols + cat_cols].copy()
X_val = val[numeric_cols + cat_cols].copy()
mlca_runtime.fit_record(preprocessor, X_train, 'train')
X_val_trans = preprocessor.transform(X_val)
print(f"Train shape {X_train.shape}, transformed val shape {getattr(X_val_trans, 'shape', 'sparse')}")
mlca_runtime.record_metric("n_features_out", float(X_val_trans.shape[1]), "val")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
