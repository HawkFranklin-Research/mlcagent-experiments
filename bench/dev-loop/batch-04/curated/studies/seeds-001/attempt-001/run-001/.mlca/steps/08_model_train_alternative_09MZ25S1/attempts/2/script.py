import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
key_cols = []
for c in all_cols:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            key_cols.append(c)
drop.update(key_cols)
feature_cols = [c for c in all_cols if c not in drop]

with open(inputs['05_preprocess_fit_preprocessing_FVVN700K/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_FW7XY320/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def get_feature_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

expected_cols = get_feature_names(preprocessor) or get_feature_names(selector) or feature_cols
expected_cols = [c for c in expected_cols if c in train.columns]
X_train = train[expected_cols].copy()
X_val = val[expected_cols].copy()

if isinstance(selector, Pipeline) and 'preprocessor' in selector.named_steps:
    frontend = selector
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

results = []
best_k = None
best_sil = -1
best_pipe = None

for k in range(2, 11):
    pipe = Pipeline([('frontend', frontend), ('kmeans', MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10))])
    fitted = mlca_runtime.fit_record(pipe, X_train, partition='train')
    labels = fitted.predict(X_train)
    X_trans = fitted.named_steps['frontend'].transform(X_train)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    sil = silhouette_score(X_trans, labels, sample_size=min(10000, len(X_train)), random_state=42)
    db = davies_bouldin_score(X_trans, labels)
    ch = calinski_harabasz_score(X_trans, labels)
    aris = []
    for b in range(5):
        weights = np.random.multinomial(len(X_train), np.ones(len(X_train))/len(X_train))
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        km_fitted = mlca_runtime.fit_record(km, X_trans, partition='train', sample_weight=weights)
        boot_labels = km_fitted.predict(X_trans)
        aris.append(adjusted_rand_score(labels, boot_labels))
    stability = np.mean(aris)
    results.append({'k': k, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch, 'stability': stability})
    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_pipe = fitted

mlca_runtime.record_metric("silhouette", best_sil, "train")
mlca_runtime.save_model(best_pipe, "alternative_model.pkl")

print(f"Best k: {best_k}, silhouette: {best_sil:.4f}")
print("Saved alternative_model.pkl")
