import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.frozen import FrozenEstimator

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
if 'targets' not in protocol:
    raise KeyError(f"missing 'targets'; available top-level keys: {list(protocol.keys())}")
if 'excluded_columns' not in protocol:
    raise KeyError(f"missing 'excluded_columns'; available top-level keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_63H1R042/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

def is_key_like(s):
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s):
        u = s.dropna().unique()
        if len(u) <= 1 or set(u).issubset({0, 1}):
            return False
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feat_cols = [c for c in train.columns if c not in targets and c not in excluded]
if group is not None and group in train.columns:
    feat_cols = [c for c in feat_cols if c != group]
feat_cols = [c for c in feat_cols if not is_key_like(train[c])]

X_train = train[feat_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

Xt = preprocessor.transform(X_train.head(50))
n_features = int(Xt.shape[1])
k = max(1, min(20, n_features))

pipe = Pipeline([
    ('preprocessing', FrozenEstimator(preprocessor)),
    ('selector', SelectKBest(mutual_info_classif, k=k)),
])
if n_features <= 300:
    try:
        pipe.set_output(transform='pandas')
    except Exception:
        pass

mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipe, 'selector.pkl')
mlca_runtime.record_metric('n_features_input', float(n_features), 'train')
mlca_runtime.record_metric('k_selected', float(k), 'train')
print(f'features={n_features} k={k} rows={len(X_train)}')
print('saved selector.pkl')
