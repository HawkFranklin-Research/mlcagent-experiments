import json
import numpy as np
import pandas as pd
import mlca_runtime
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise ValueError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
n = len(train)
exclude = set(targets) | excluded
if group:
    exclude.add(group)

features = []
for c in train.columns:
    if c in exclude:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        try:
            if n > 1 and s.nunique(dropna=False) == n:
                continue
        except TypeError:
            pass
    elif pd.api.types.is_integer_dtype(s):
        if n > 1 and s.nunique(dropna=False) == n and s.min() >= 0:
            continue
    features.append(c)

datetime_cols, numeric_cols, cat_low, cat_high = [], [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun > 5 and nun > 0.05 * n:
            cat_high.append(c)
        else:
            cat_low.append(c)

placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in features:
    if c in cat_high or c in datetime_cols:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        if vals.isin(placeholder_tokens - {''}).any() or (vals == '').any():
            ph_cols.append(c)
print(f"Placeholder tokens found in {len(ph_cols)} text columns")

num_log, num_plain = [], []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 0 and s.min() >= 0 and s.skew() > 2:
        num_log.append(c)
    else:
        num_plain.append(c)

est_w = len(num_plain) + len(num_log) + len(datetime_cols)
for c in cat_low:
    est_w += min(train[c].nunique(dropna=False) + 1, 500)
use_dense = est_w <= 300


def make_num_pipe(log):
    steps = []
    if log:
        steps.append(('log', FunctionTransformer(np.log1p)))
    steps.append(('imp', SimpleImputer(strategy='median')))
    steps.append(('sc', StandardScaler()))
    return Pipeline(steps)


cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=not use_dense)),
])

dt_pipe = Pipeline([
    ('to_dt', FunctionTransformer(partial(np.asarray, dtype='datetime64[D]'))),
    ('to_float', FunctionTransformer(partial(np.ndarray.astype, dtype=np.float64))),
])

transformers = []
if num_plain:
    transformers.append(('num', make_num_pipe(False), num_plain))
if num_log:
    transformers.append(('num_log', make_num_pipe(True), num_log))
if cat_low:
    transformers.append(('cat', cat_pipe, cat_low))
if datetime_cols:
    transformers.append(('dt', dt_pipe, datetime_cols))

ct = ColumnTransformer(transformers, remainder='drop')
if use_dense:
    ct.set_output(transform='pandas')

preprocessor = Pipeline([
    ('strip', FunctionTransformer(partial(pd.DataFrame.replace, to_replace=r'^\s+|\s+$', value='', regex=True))),
    ('ph', FunctionTransformer(partial(pd.DataFrame.replace, to_replace={'?': np.nan, '-': np.nan, 'NA': np.nan, 'N/A': np.nan, '': np.nan}))),
    ('ct', ct),
])

final_features = [c for c in features if c not in cat_high]
X = train[final_features]
fitted = mlca_runtime.fit_record(preprocessor, X, 'train')
mlca_runtime.save_model(fitted, "preprocessor.pkl")
out = fitted.transform(X)
print(f"fitted feature cols={len(final_features)} output cols={out.shape[1]} dense={use_dense}")
