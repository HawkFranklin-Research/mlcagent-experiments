import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

for k in ('targets', 'excluded_columns', 'task'):
    if k not in protocol:
        raise KeyError(f"Missing protocol key '{k}'. Available: {sorted(protocol.keys())}")

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column', None)

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
candidates = [c for c in train.columns if c not in drop_set]

# Key-like: text codes / contiguous integer ids with all distinct values
key_like = []
for c in candidates:
    s = train[c]; n = len(s)
    if n <= 1 or s.nunique(dropna=False) < n:
        continue
    if s.dtype == object:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = np.sort(s.dropna().values)
        if len(vals) > 1 and (vals[-1] - vals[0] + 1) == len(vals):
            key_like.append(c)
candidates = [c for c in candidates if c not in set(key_like)]

# Derivation check: exact equality with a target (row-sum check implies equality)
derived = []
for f in candidates:
    for t in targets:
        if train[f].equals(train[t]):
            derived.append(f)
            print(f"Excluded feature {f}: exact equality with target {t}")
            break
candidates = [c for c in candidates if c not in set(derived)]

numeric_cols, cat_cols, dt_cols, text_cols = [], [], [], []
for c in candidates:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        nun = s.nunique(dropna=True)
        (text_cols if nun > max(20, 0.05 * len(train)) else cat_cols).append(c)
    elif pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_bool_dtype(s):
            numeric_cols.append(c)
        elif pd.api.types.is_integer_dtype(s):
            uniq = set(s.dropna().unique()); nun = s.nunique(dropna=True)
            (numeric_cols if (uniq == {0, 1} or nun > 10) else cat_cols).append(c)
        else:
            numeric_cols.append(c)

# Placeholder detection
placeholders = {'?', '-', 'NA', 'N/A'}
ph_cols = []
for c in cat_cols + text_cols:
    s = train[c].dropna().astype(str)
    if len(s) and (s.isin(placeholders).any() or (s.str.strip() == '').any() or (s != s.str.strip()).any()):
        ph_cols.append(c)
print(f"Placeholder columns: {len(ph_cols)}")

for c in text_cols:
    print(f"Dropped high-cardinality text feature {c}")
candidates = [c for c in candidates if c not in set(text_cols)]
cat_cols = [c for c in cat_cols if c not in set(text_cols)]

# Numeric correlation warning (numeric outcomes only)
for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]):
        for c in numeric_cols:
            try:
                corr = train[[c, t]].corr().iloc[0, 1]
                if abs(corr) >= 0.98:
                    print(f"Warning: {c} |corr|={abs(corr):.3f} with {t}")
            except Exception:
                pass

# Datetime handling
for c in list(dt_cols):
    dt = pd.to_datetime(train[c], errors='coerce')
    for part in ('hour', 'dayofweek', 'month'):
        train[f"{c}_{part}"] = getattr(dt.dt, part)
        numeric_cols.append(f"{c}_{part}")
    candidates = [x for x in candidates if x != c]
for c in list(cat_cols):
    try:
        dt = pd.to_datetime(train[c], errors='coerce')
        if dt.notna().sum() > 0.9 * len(train):
            for part in ('hour', 'dayofweek', 'month'):
                train[f"{c}_{part}"] = getattr(dt.dt, part)
                numeric_cols.append(f"{c}_{part}")
            candidates = [x for x in candidates if x != c]
            cat_cols.remove(c)
    except Exception:
        pass

numeric_cols = [c for c in numeric_cols if c in candidates]
cat_cols = [c for c in cat_cols if c in candidates]

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
parts = [('num', num_pipe, numeric_cols)]
if cat_cols:
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    parts.append(('cat', cat_pipe, cat_cols))

pre = ColumnTransformer(parts, remainder='drop')
n_out = len(numeric_cols) + sum(train[c].nunique(dropna=True) + (1 if train[c].isna().any() else 0) for c in cat_cols)
if n_out <= 300:
    pre.set_output(transform='pandas')

feature_cols = numeric_cols + cat_cols
X_train = train[feature_cols].copy()
mlca_runtime.fit_record(pre, X_train, 'train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')
print(f"Saved preprocessor: {len(feature_cols)} features, ~{n_out} output cols")
