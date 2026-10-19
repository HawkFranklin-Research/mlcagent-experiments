import mlca_runtime
import json, re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths['_protocol'].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

targets = list(need(proto, 'targets') or [])
excluded = list(need(proto, 'excluded_columns') or [])
group = need(proto, 'group_column')
task = need(proto, 'task')
metric = need(task, 'primary_metric')
train = pd.read_parquet(paths['train'])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return len(s) > 1 and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return bool(re.search(r'(^|_)(id|key|code|seq|index)(_|$)', str(s.name), re.I)) and len(s) > 1 and s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        date_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) <= max(5, int(0.05 * len(train))):
            cat_cols.append(c)
    elif pd.api.types.is_bool_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
        num_cols.append(c)
if date_cols:
    raise ValueError(f"Date columns need explicit expansion: {date_cols}")

feature_cols = num_cols + cat_cols
transforms = []
if num_cols:
    transforms.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
if cat_cols:
    transforms.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols))
pre = ColumnTransformer(transforms, remainder='drop')
pre.set_output(transform='pandas')

X = train[feature_cols]
pre = mlca_runtime.fit_record(pre, X, 'train')
X_proc = pre.transform(X)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
full = Pipeline([('preprocessor', pre), ('mlp', mlp)])
full = mlca_runtime.fit_record(full, X, 'train', y=X_proc)

pred = full.predict(X)
mse = float(np.mean((pred - X_proc.to_numpy()) ** 2))
mlca_runtime.save_model(full, 'alternative_model.pkl')
mlca_runtime.record_metric(metric, mse, 'train')
mlca_runtime.record_metric('embedding_dim', 8, 'train')
print(f"rows={len(train)} features={len(feature_cols)}")
print(f"{metric}={mse:.6f}")
