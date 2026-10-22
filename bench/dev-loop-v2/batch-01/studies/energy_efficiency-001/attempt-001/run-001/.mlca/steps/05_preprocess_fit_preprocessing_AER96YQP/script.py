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
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise ValueError(f"Missing key '{k}' in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

n = len(train)
key_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == 'object' or str(s.dtype) == 'category':
        if s.nunique() == n:
            key_cols.append(c)
    elif np.issubdtype(s.dtype, np.integer) and s.nunique() == n:
        key_cols.append(c)
feature_cols = [c for c in feature_cols if c not in key_cols]
if key_cols:
    print(f"Excluded key-like: {key_cols}")

excluded_deriv = []
for target in targets:
    y = train[target].values
    for c in feature_cols:
        if np.allclose(train[c].values, y, equal_nan=False):
            excluded_deriv.append((c, target, 'exact'))
    nonneg = [c for c in feature_cols if (train[c] >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(nonneg, r):
            if np.allclose(train[list(combo)].sum(axis=1).values, y, equal_nan=False):
                for c in combo:
                    excluded_deriv.append((c, target, f'sum of {combo}'))
                break
        else:
            continue
        break
excluded_deriv = list(set(excluded_deriv))
for c, t, reason in excluded_deriv:
    print(f"Excluded derivation: {c} -> {t} ({reason})")
feature_cols = [c for c in feature_cols if c not in [e[0] for e in excluded_deriv]]

corr_warn = []
for target in targets:
    if np.issubdtype(train[target].dtype, np.number):
        for c in feature_cols:
            if np.issubdtype(train[c].dtype, np.number):
                corr = np.corrcoef(train[c].fillna(train[c].median()), train[target].fillna(train[target].median()))[0,1]
                if abs(corr) >= 0.98:
                    corr_warn.append(c)
if corr_warn:
    print(f"Correlation warning >=0.98: {corr_warn}")

text_cols = [c for c in feature_cols if train[c].dtype == 'object' or str(train[c].dtype) == 'category']
placeholders = ['?', '-', 'NA', 'N/A']
ph_found = {}
for c in text_cols:
    s = train[c].astype(str).str.strip()
    cnt = sum((s == p).sum() for p in placeholders) + (s == '').sum()
    if cnt:
        ph_found[c] = cnt
if ph_found:
    print(f"Placeholders found: {ph_found}")

numeric_cols, cat_cols, high_card, date_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if np.issubdtype(s.dtype, np.datetime64):
        date_cols.append(c)
    elif np.issubdtype(s.dtype, np.number):
        numeric_cols.append(c)
    elif s.dtype == 'object' or str(s.dtype) == 'category':
        try:
            pd.to_datetime(s.dropna().head(5))
            date_cols.append(c)
        except Exception:
            if s.nunique() > 0.05 * n:
                high_card.append(c)
            else:
                cat_cols.append(c)
    else:
        numeric_cols.append(c)

if high_card:
    print(f"Dropping high-card text: {high_card}")
feature_cols = [c for c in feature_cols if c not in high_card and c not in date_cols]

def build_preprocessor(use_pandas=False):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    if use_pandas:
        num_pipe.set_output(transform='pandas')
        cat_pipe.set_output(transform='pandas')
    transformers = []
    if numeric_cols:
        transformers.append(('num', num_pipe, numeric_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipe, cat_cols))
    ct = ColumnTransformer(transformers, remainder='drop')
    if use_pandas:
        ct.set_output(transform='pandas')
    return ct

X_train = train[feature_cols]
pre = build_preprocessor(use_pandas=False)
mlca_runtime.fit_record(pre, X_train, partition='train', y=None)
width = len(pre.get_feature_names_out())
if width <= 300:
    pre = build_preprocessor(use_pandas=True)
    mlca_runtime.fit_record(pre, X_train, partition='train', y=None)

mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"Saved preprocessor: {len(feature_cols)} features, width {width}")
