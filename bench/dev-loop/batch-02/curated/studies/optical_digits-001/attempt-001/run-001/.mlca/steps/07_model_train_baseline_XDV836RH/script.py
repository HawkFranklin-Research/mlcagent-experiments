import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_01V2VW4T/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def contains(pipe, obj):
    if not isinstance(pipe, Pipeline):
        return False
    return any(step is obj for _, step in pipe.steps)

def raw_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

if contains(selector, preprocessor):
    frontend = selector
    raw_features = raw_cols(selector)
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
    raw_features = raw_cols(preprocessor)

if raw_features is None:
    raw_features = list(train.columns)
raw_features = [c for c in raw_features if c not in drop]

def n_out(obj):
    if hasattr(obj, 'n_features_'):
        return int(obj.n_features_)
    if hasattr(obj, 'k'):
        return int(obj.k)
    if isinstance(obj, Pipeline):
        return n_out(obj.steps[-1][1])
    return None

n_feat = n_out(selector) or n_out(preprocessor) or len(raw_features)
n_components = min(8, n_feat)

pipeline = Pipeline([
    ('frontend', frontend),
    ('pca', PCA(n_components=n_components, random_state=42)),
])

X_train = train[raw_features]
fitted = mlca_runtime.fit_record(pipeline, X_train, partition='train')

def recon_err(pipe, X):
    front = pipe.named_steps['frontend'].transform(X)
    pca = pipe.named_steps['pca']
    rec = pca.inverse_transform(pca.transform(front))
    return float(np.mean((front - rec) ** 2))

tr = recon_err(fitted, X_train)
va = recon_err(fitted, val[raw_features])

mlca_runtime.record_metric('reconstruction_error', tr, split='train')
mlca_runtime.record_metric('reconstruction_error', va, split='val')
mlca_runtime.save_model(fitted, 'model.pkl')

print(f"train {train.shape} val {val.shape}")
print(f"features {len(raw_features)} pca {n_components}")
print(f"train_recon {tr:.4f} val_recon {va:.4f}")
