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
protocol = json.loads(inputs["_protocol"].read_text())
top = sorted(protocol.keys())

def get(k, default=None, required=True):
    if k in protocol:
        return protocol[k]
    if required:
        raise KeyError(f"Missing {k}. Available top-level keys: {top}")
    return default

targets = get("targets")
excluded = list(get("excluded_columns", []))
group = get("group_column", None, required=False)
task = get("task", required=False) or {}

train = pd.read_parquet(inputs["train"])

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in train.columns:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].copy()
        str_s = s.astype(str).str.strip()
        mask = str_s.isin(ph_tokens)
        if mask.any():
            ph_cols.append(c)
        s = str_s.mask(mask, np.nan)
        s = s.replace({'nan': np.nan, 'None': np.nan})
        train[c] = s
print(f"placeholder_cols={len(ph_cols)} names={ph_cols[:5]}")

all_cols = list(train.columns)
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in all_cols if c not in drop]

drop_keys = []
for c in list(feature_cols):
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
        if pd.api.types.is_integer_dtype(s) and not s.is_monotonic_increasing:
            continue
        drop_keys.append(c)
feature_cols = [c for c in feature_cols if c not in drop_keys]
if drop_keys:
    print(f"dropped_key_like={drop_keys[:5]}")

deriv = []
for t in targets:
    if t not in train.columns:
        continue
    for c in list(feature_cols):
        if pd.api.types.is_string_dtype(train[c]) and pd.api.types.is_string_dtype(train[t]):
            if (train[c].astype(str) == train[t].astype(str)).all():
                deriv.append(c)
                feature_cols.remove(c)
                break
        elif pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(train[t]):
            if np.allclose(train[c], train[t], equal_nan=True):
                deriv.append(c)
                feature_cols.remove(c)
                break
if deriv:
    print(f"excluded_derivations={deriv[:5]}")

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols and train[c].nunique(dropna=False) <= 0.05 * len(train)]

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)

transformers = []
if num_cols:
    transformers.append(('num', num_pipe, num_cols))
if cat_cols:
    if width <= 300:
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    else:
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    transformers.append(('cat', cat_pipe, cat_cols))

ct = ColumnTransformer(transformers, remainder='drop')
if width <= 300:
    ct.set_output(transform='pandas')

mlca_runtime.fit_record(ct, train[feature_cols], "train")
mlca_runtime.save_model(ct, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_numeric", len(num_cols), "train")
mlca_runtime.record_metric("n_categorical", len(cat_cols), "train")
print(f"features={len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)}")
