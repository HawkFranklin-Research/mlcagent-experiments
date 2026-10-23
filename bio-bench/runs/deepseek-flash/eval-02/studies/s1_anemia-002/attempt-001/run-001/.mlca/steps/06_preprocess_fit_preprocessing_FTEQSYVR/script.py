import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = protocol['targets']
excluded = protocol.get('excluded_columns') or []
group = protocol.get('group_column')
target_col = targets[0]
n = len(train)

drop_cols = set(targets) | set(excluded)
if group and group in train.columns:
    drop_cols.add(group)
feature_cols = [c for c in train.columns if c not in drop_cols]

key_like = [c for c in feature_cols
            if train[c].nunique(dropna=True) == n
            and (train[c].dtype == object or pd.api.types.is_string_dtype(train[c])
                 or pd.api.types.is_integer_dtype(train[c]))]
for c in key_like:
    feature_cols.remove(c)
if key_like:
    print(f"Key-like dropped: {key_like}")

derived = []
for c in list(feature_cols):
    try:
        if train[c].equals(train[target_col]):
            derived.append((c, 'exact target match'))
    except Exception:
        pass
if pd.api.types.is_numeric_dtype(train[target_col]):
    tgt = train[target_col].astype(float).values
    used = {c for c, _ in derived}
    numc = [c for c in feature_cols if c not in used and pd.api.types.is_numeric_dtype(train[c])]
    found = False
    if len(numc) <= 40:
        for k in (1, 2, 3):
            for combo in combinations(numc, k):
                sub = train[list(combo)].fillna(0)
                if (sub < 0).any().any():
                    continue
                if np.allclose(sub.sum(axis=1).values, tgt, atol=1e-8):
                    derived.extend([(c, f'row-sum {combo}') for c in combo])
                    found = True
                    break
            if found:
                break
for c, reason in derived:
    if c in feature_cols:
        feature_cols.remove(c)
        print(f"Derived dropped {c}: {reason}")

tokens = {'?', '-', 'NA', 'N/A'}
placeholder_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if (s.notna() & stripped.isin(tokens | {''})).any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders in: {placeholder_cols}")
    for df in (train, val):
        for c in placeholder_cols:
            s = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[c] = s.where(~s.isin(tokens | {''}), np.nan)

date_cols = []
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
        continue
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        try:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() > 0.9:
                date_cols.append(c)
        except Exception:
            pass
for c in date_cols:
    for df in (train, val):
        p = pd.to_datetime(df[c], errors='coerce')
        df[c + '_hour'] = p.dt.hour.fillna(-1).astype(int)
        df[c + '_dow'] = p.dt.dayofweek.fillna(-1).astype(int)
        df[c + '_month'] = p.dt.month.fillna(-1).astype(int)
    feature_cols.remove(c)
    feature_cols.extend([c + '_hour', c + '_dow', c + '_month'])
if date_cols:
    print(f"Dates parsed: {date_cols}")

for c in list(feature_cols):
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) > 0.05 * n:
        feature_cols.remove(c)

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s):
        train[c] = s.astype(int)
        val[c] = val[c].astype(int)
        numeric_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in categorical_cols)

num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
if est_width <= 300:
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    set_pandas = True
else:
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    set_pandas = False

transformers = []
if numeric_cols:
    transformers.append(('num', num_pipe, numeric_cols))
if categorical_cols:
    transformers.append(('cat', cat_pipe, categorical_cols))
preprocessor = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
if set_pandas:
    preprocessor.set_output(transform='pandas')

X_train = train[feature_cols].copy()
mlca_runtime.fit_record(preprocessor, X_train, 'train')
Xval = preprocessor.transform(val[feature_cols].copy())
print(f"Features {len(feature_cols)} -> encoded {Xval.shape[1]}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
