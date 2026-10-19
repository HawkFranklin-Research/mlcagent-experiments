import mlca_runtime
import json, itertools
import numpy as np
import pandas as pd
from operator import attrgetter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
y = train[targets[0]]

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

# Derivation check
derived = {}
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and (s == y).all():
        derived[c] = 'exact-match'
non_neg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].min() >= 0]
for r in (2, 3):
    for combo in itertools.combinations(non_neg, r):
        if (train[list(combo)].sum(axis=1) == y).all():
            for c in combo:
                derived.setdefault(c, 'sum-of-up-to-3-nonneg')

warn = []
for c in feature_cols:
    if c in derived:
        continue
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        corr = s.corr(y)
        if pd.api.types.is_float(corr) and abs(corr) >= 0.98:
            warn.append(c)
if warn:
    print(f"Warn |corr|>=0.98: {warn}")
for c, reason in derived.items():
    print(f"Exclude {c}: {reason}")
feature_cols = [c for c in feature_cols if c not in derived]

# Placeholder detection
ph_count = 0
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        vals = train[c].astype(str).str.strip()
        if vals.isin(['?', '-', 'NA', 'N/A', '']).any():
            ph_count += 1
if ph_count:
    print(f"Placeholders in {ph_count} features -> NaN")

# Classify columns
num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        u = set(s.dropna().unique())
        if u and u.issubset({0, 1, 0.0, 1.0}):
            num_cols.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() <= 50:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        try:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() > 0.9:
                date_cols.append(c)
                continue
        except Exception:
            pass
        if s.nunique() <= 0.05 * len(train):
            cat_cols.append(c)

# Preprocessing pipeline
ph_kw = {'to_replace': [r'^\s*\?\s*$', r'^\s*-\s*$', r'^\s*NA\s*$', r'^\s*N/A\s*$', r'^\s*$'],
         'value': np.nan, 'regex': True}

transformers = []
if num_cols:
    transformers.append(('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))
for dc in date_cols:
    for part, getter_name in (('month', 'dt.month'), ('dow', 'dt.dayofweek'), ('day', 'dt.day')):
        transformers.append((
            f'{dc}_{part}',
            Pipeline([
                ('parse', FunctionTransformer(pd.to_datetime)),
                ('extract', FunctionTransformer(attrgetter(getter_name))),
                ('to2d', FunctionTransformer(pd.Series.to_frame, kw_args={'name': f'{dc}_{part}'})),
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler()),
            ]),
            dc,
        ))

ct = ColumnTransformer(transformers, remainder='drop').set_output(transform='pandas')
preprocessor = Pipeline([
    ('placeholder', FunctionTransformer(pd.DataFrame.replace, kw_args=ph_kw)),
    ('encode', ct),
])
preprocessor.set_output(transform='pandas')

mlca_runtime.fit_record(preprocessor, train[feature_cols], 'train')
X_t = preprocessor.transform(train[feature_cols])
mlca_runtime.record_metric('n_features_in', float(len(feature_cols)), 'train')
mlca_runtime.record_metric('n_features_out', float(X_t.shape[1]), 'train')

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
print(f"Preprocessor fitted: {len(feature_cols)} -> {X_t.shape[1]} features")
