import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')
task = proto['task']

base_cols = [c for c in train.columns if c not in targets and c not in excluded and (group is None or c != group)]
feature_cols, key_like = [], []
for c in base_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c); continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s) and s.is_monotonic_increasing:
            key_like.append(c); continue
    feature_cols.append(c)

derived = {}
for t in targets:
    y = train[t]
    for c in feature_cols:
        if c == t: continue
        s = train[c]
        if s.isna().equals(y.isna()) and (s[s.notna()] == y[y.notna()]).all():
            derived[c] = f"exact match with {t}"
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.notna().all():
        nf = [c for c in feature_cols if c not in derived and pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]) and train[c].notna().all() and (train[c] >= 0).all()]
        for r in range(1, 4):
            for combo in combinations(nf, r):
                if any(c in derived for c in combo): continue
                s = train[list(combo)].sum(axis=1)
                if np.allclose(s.values, y.values, rtol=1e-9, atol=1e-12):
                    for c in combo: derived[c] = f"sum {combo} equals {t}"
feature_cols = [c for c in feature_cols if c not in derived]

warn = []
for t in targets:
    y = train[t]
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]):
                if abs(train[c].corr(y)) >= 0.98:
                    warn.append(c)

text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
ph = {p: 0 for p in ['?', '-', 'NA', 'N/A']}
ws = 0
for c in text_cols:
    s = train[c].dropna().astype(str).str.strip()
    for p in ph:
        if (s == p).any(): ph[p] += 1
    if (s == '').any(): ws += 1
print(f"Placeholders: {ph}, whitespace-only cols={ws}")
if key_like: print(f"Key-like excluded: {key_like}")
if derived: print(f"Target-derived excluded: {derived}")
if warn: print(f"Warning |corr|>=0.98 kept: {warn}")

n = len(train)
numeric_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * n:
            drop_cols.append(c)
        else:
            cat_cols.append(c)

est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) + 1 for c in cat_cols)
pandas_output = est_width <= 300
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=not pandas_output)

transformers = []
if numeric_cols:
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    if pandas_output: num_pipe.set_output(transform='pandas')
    transformers.append(('num', num_pipe, numeric_cols))
if cat_cols:
    strip = FunctionTransformer(func=pd.DataFrame.replace, kw_args={'to_replace': r'^\s+|\s+$', 'value': '', 'regex': True}, feature_names_out='one-to-one')
    to_nan = FunctionTransformer(func=pd.DataFrame.replace, kw_args={'to_replace': ['?', '-', 'NA', 'N/A', ''], 'value': np.nan, 'regex': False}, feature_names_out='one-to-one')
    cat_pipe = Pipeline([('strip', strip), ('to_nan', to_nan), ('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', onehot)])
    if pandas_output: cat_pipe.set_output(transform='pandas')
    transformers.append(('cat', cat_pipe, cat_cols))

ct = ColumnTransformer(transformers, remainder='drop')
if pandas_output: ct.set_output(transform='pandas')

X_train = train[feature_cols]
X_val = val[feature_cols]
preprocessor = mlca_runtime.fit_record(ct, X_train, 'train')
X_val_t = preprocessor.transform(X_val)
print(f"Train transformed: {preprocessor.transform(X_train).shape}; Val transformed: {X_val_t.shape}")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_transformed_columns", X_val_t.shape[1], "val")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
