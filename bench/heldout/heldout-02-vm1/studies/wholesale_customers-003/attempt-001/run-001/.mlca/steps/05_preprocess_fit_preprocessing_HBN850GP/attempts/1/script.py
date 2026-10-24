import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
train = pd.read_parquet(paths['train'])

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key '{k}'. Available top-level keys: {sorted(d)}")
    return d[k]

targets = req(protocol, 'targets')
excluded = req(protocol, 'excluded_columns')
task = req(protocol, 'task')
group_col = protocol.get('group_column', None)
if group_col is not None and group_col not in train.columns:
    group_col = None

exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)
candidate = [c for c in train.columns if c not in exclude]

def key_like(s):
    if s.nunique(dropna=True) != len(train):
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)

key_cols = [c for c in candidate if key_like(train[c])]
features = [c for c in candidate if c not in key_cols]

placeholder_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        s2 = s.astype('string').str.strip()
        if s2.isin(['?', '-', 'NA', 'N/A', '']).any():
            placeholder_cols.append(c)

num_cols, heavy_cols, low_cat_num, low_cat_text, high_text = [], [], [], [], []
card_limit = max(5, int(0.05 * len(train)))
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        try:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() > 0.9:
                continue
        except Exception:
            pass
        if s.nunique(dropna=True) <= card_limit:
            low_cat_text.append(c)
        else:
            high_text.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        nun = s.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(s) and 2 < nun <= card_limit:
            low_cat_num.append(c)
        else:
            num_cols.append(c)
            if s.min() >= 0 and s.skew() > 1.0:
                heavy_cols.append(c)

heavy_set = set(heavy_cols)
num_regular = [c for c in num_cols if c not in heavy_set]
est_width = len(num_regular) + len(heavy_cols) + sum(train[c].nunique(dropna=True) for c in low_cat_num + low_cat_text)
use_pandas = est_width <= 300

def make_onehot():
    return OneHotEncoder(handle_unknown='ignore', sparse_output=not use_pandas)

transformers = []
if num_regular:
    transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_regular))
if heavy_cols:
    transformers.append(('heavy', Pipeline([('imp', SimpleImputer(strategy='median')), ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')), ('sc', StandardScaler())]), heavy_cols))
if low_cat_num:
    transformers.append(('cat_num', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', make_onehot())]), low_cat_num))
if low_cat_text:
    steps = [
        ('fill_na', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='__MISSING__')),
        ('strip', FunctionTransformer(np.char.strip, feature_names_out='one-to-one')),
        ('restore_na', SimpleImputer(missing_values='__MISSING__', strategy='constant', fill_value=np.nan)),
    ]
    for i, tok in enumerate(['?', '-', 'NA', 'N/A', '']):
        steps.append((f'miss_{i}', SimpleImputer(missing_values=tok, strategy='constant', fill_value=np.nan)))
    steps += [('imp', SimpleImputer(strategy='most_frequent')), ('oh', make_onehot())]
    transformers.append(('cat_text', Pipeline(steps), low_cat_text))

preprocessor = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
if use_pandas:
    preprocessor.set_output(transform='pandas')

preprocessor = mlca_runtime.fit_record(preprocessor, train[features], 'train')
mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
mlca_runtime.record_metric('n_features', len(features), 'train')

print(f"Features: {len(features)}; key-like dropped: {len(key_cols)}; targets: {len(targets)}")
print(f"Placeholder tokens in {len(placeholder_cols)} columns")
print(f"Heavy-tailed log1p: {len(heavy_cols)} columns")
print(f"One-hot categorical: {len(low_cat_num)+len(low_cat_text)}; high-cardinality dropped: {len(high_text)}")
print(f"Estimated output width: {est_width}")
