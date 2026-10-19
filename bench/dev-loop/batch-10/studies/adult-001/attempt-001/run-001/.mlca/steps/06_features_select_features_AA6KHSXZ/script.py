import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
target = targets[0] if isinstance(targets, list) else targets
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_F9B0NJ26/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)

if hasattr(preproc, 'feature_names_in_') and preproc.feature_names_in_ is not None:
    feature_cols = list(preproc.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    for c in list(feature_cols):
        if train[c].dtype == 'object' or pd.api.types.is_string_dtype(train[c]):
            if train[c].nunique() == len(train):
                feature_cols.remove(c)
        elif pd.api.types.is_integer_dtype(train[c]):
            if train[c].nunique() == len(train) and (c.lower().endswith('id') or c.lower().endswith('key')):
                feature_cols.remove(c)

for c in targets + list(excluded) + ([group] if group else []):
    if c in feature_cols:
        raise ValueError(f"Column {c} must not be a feature")

X_train = train[feature_cols]
y_train = train[target]

n_features_pre = None
try:
    if hasattr(preproc, 'transform'):
        out = preproc.transform(X_train.head(50))
        n_features_pre = out.shape[1]
except Exception:
    pass

if n_features_pre is not None:
    k = min(20, n_features_pre)
    selector = SelectKBest(mutual_info_classif, k=k)
else:
    selector = SelectKBest(mutual_info_classif, k='all')

pipeline = Pipeline([
    ('preprocessor', preproc),
    ('selector', selector)
])

fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)

sel = fitted.named_steps['selector']
if sel.k == 'all':
    sel.k = min(20, len(sel.scores_))

mlca_runtime.save_model(fitted, 'selector.pkl')

n_all = len(sel.scores_)
k_final = sel.k
mlca_runtime.record_metric('n_features_preprocessed', n_all, 'train')
mlca_runtime.record_metric('n_features_selected', k_final, 'train')

print(f"Train shape: {train.shape}, raw features: {len(feature_cols)}")
print(f"Preprocessed features: {n_all}, selected: {k_final}")
