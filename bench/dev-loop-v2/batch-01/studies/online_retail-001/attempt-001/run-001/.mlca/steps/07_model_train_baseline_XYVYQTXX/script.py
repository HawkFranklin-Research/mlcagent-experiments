import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = proto.get('targets', [])
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column', None)
task = proto.get('task', {})

with open(inputs['05_preprocess_fit_preprocessing_8WCJT1S7/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)
with open(inputs['06_features_retain_features_2NP0WC88/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def get_input_columns(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

preproc_cols = get_input_columns(preproc)
selector_cols = get_input_columns(selector)
raw_cols = list(train.columns)

if selector_cols is not None and set(selector_cols).issubset(set(raw_cols)):
    frontend = selector
    input_cols = selector_cols
else:
    frontend = Pipeline([('preproc', preproc), ('selector', selector)])
    input_cols = preproc_cols if preproc_cols is not None else raw_cols

missing = [c for c in input_cols if c not in raw_cols]
if missing:
    raise ValueError(f"Missing input columns: {missing}")
X_train = train[input_cols].copy()

k_range = range(2, 9)
results = []
n = len(X_train)
for k in k_range:
    if k >= n:
        continue
    pipe = Pipeline([
        ('frontend', frontend),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipe = mlca_runtime.fit_record(pipe, X_train, partition='train')
    labels = pipe.named_steps['kmeans'].labels_
    if len(np.unique(labels)) < 2:
        continue
    X_trans = pipe.named_steps['frontend'].transform(X_train)
    sil = silhouette_score(X_trans, labels, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(X_trans, labels)
    ch = calinski_harabasz_score(X_trans, labels)
    aris = []
    for _ in range(3):
        weights = np.random.multinomial(n, np.ones(n)/n)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km = mlca_runtime.fit_record(km, X_trans, partition='train', sample_weight=weights)
        aris.append(adjusted_rand_score(labels, km.labels_))
    stab = np.mean(aris)
    results.append({'k': k, 'sil': sil, 'db': db, 'ch': ch, 'stab': stab, 'pipe': pipe})

if not results:
    raise RuntimeError("No valid clustering candidate")

best = max(results, key=lambda r: (r['sil'], r['stab']))
best_pipe = best['pipe']
mlca_runtime.save_model(best_pipe, 'model.pkl')

mlca_runtime.record_metric('silhouette', float(best['sil']), 'train')
mlca_runtime.record_metric('n_clusters', float(best['k']), 'train')
mlca_runtime.record_metric('stability_ari', float(best['stab']), 'train')
print(f"Best k={best['k']}, silhouette={best['sil']:.4f}, stability_ari={best['stab']:.4f}")
print(f"Train size: {n}, feature columns: {len(input_cols)}")
