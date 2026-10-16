import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol_key(key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key('targets')
excluded = get_protocol_key('excluded_columns')
group = get_protocol_key('group_column')
task = get_protocol_key('task')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_01V2VW4T/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

with open(inputs['05_preprocess_fit_preprocessing_78KYAZ91/manifest.json'], 'r') as f:
    prep_manifest = json.load(f)
with open(inputs['06_features_retain_features_01V2VW4T/manifest.json'], 'r') as f:
    sel_manifest = json.load(f)

def get_raw_features(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

raw_features = get_raw_features(selector) or get_raw_features(preprocessor)
if raw_features is None:
    all_cols = list(train.columns)
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = []
    for c in all_cols:
        if c in drop:
            continue
        s = train[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            if s.nunique() == len(s):
                continue
        if pd.api.types.is_integer_dtype(s):
            if s.nunique() == len(s) and (s.is_monotonic_increasing or 'id' in c.lower()):
                continue
        feature_cols.append(c)
    raw_features = feature_cols

raw_features = [c for c in raw_features if c not in set(targets) | set(excluded) | ({group} if group else set())]

def selector_includes_preprocessor(sel, prep):
    if not isinstance(sel, Pipeline):
        return False
    for name, step in sel.steps:
        if step is prep:
            return True
        if isinstance(step, ColumnTransformer) and isinstance(prep, ColumnTransformer):
            if hasattr(step, 'feature_names_in_') and hasattr(prep, 'feature_names_in_'):
                if list(step.feature_names_in_) == list(prep.feature_names_in_):
                    return True
    return False

if selector_includes_preprocessor(selector, preprocessor):
    frontend = selector
else:
    frontend = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector)
    ])

def get_n_features_from_selector(sel):
    if hasattr(sel, 'k'):
        return sel.k
    if hasattr(sel, 'n_features_'):
        return sel.n_features_
    if isinstance(sel, Pipeline):
        last = sel.steps[-1][1]
        if hasattr(last, 'k'):
            return last.k
        if hasattr(last, 'n_features_'):
            return last.n_features_
    return None

def get_n_features_from_manifest(manifest):
    for key in ['n_features', 'n_features_out', 'num_features', 'selected_features', 'feature_names']:
        if key in manifest:
            val = manifest[key]
            if isinstance(val, int):
                return val
            if isinstance(val, list):
                return len(val)
    return None

n_features_after = (get_n_features_from_selector(selector) or
                    get_n_features_from_manifest(sel_manifest) or
                    get_n_features_from_manifest(prep_manifest) or
                    len(raw_features))
n_components = min(8, n_features_after)

pipeline = Pipeline([
    ('frontend', frontend),
    ('pca', PCA(n_components=n_components, random_state=42))
])

X_train = train[raw_features]
fitted = mlca_runtime.fit_record(pipeline, X_train, partition='train')

def reconstruction_error(pipeline, X):
    frontend = pipeline.named_steps['frontend']
    pca = pipeline.named_steps['pca']
    X_front = frontend.transform(X)
    X_pca = pca.transform(X_front)
    X_recon = pca.inverse_transform(X_pca)
    return np.mean((X_front - X_recon) ** 2)

train_err = reconstruction_error(fitted, X_train)
val_err = reconstruction_error(fitted, val[raw_features])

mlca_runtime.record_metric('reconstruction_error', float(train_err), split='train')
mlca_runtime.record_metric('reconstruction_error', float(val_err), split='val')

mlca_runtime.save_model(fitted, 'model.pkl')

print(f"Train shape: {train.shape}, Val shape: {val.shape}")
print(f"Features: {len(raw_features)}, PCA components: {n_components}")
print(f"Train reconstruction error (scaled): {train_err:.4f}")
print(f"Val reconstruction error (scaled): {val_err:.4f}")
