import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.base import clone

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing {key} in protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])

preprocessor = pickle.load(open(inputs['05_preprocess_fit_preprocessing_X6XZMNTX/preprocessor.pkl'], 'rb'))
selector = pickle.load(open(inputs['06_features_retain_features_79PZREQ1/selector_all.pkl'], 'rb'))

def get_input_columns(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = list(obj.named_steps.values())[0]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

sel_cols = get_input_columns(selector)
pre_cols = get_input_columns(preprocessor)

if sel_cols is not None and set(sel_cols).issubset(set(train.columns)):
    frontend = selector
    expected_cols = sel_cols
elif pre_cols is not None and set(pre_cols).issubset(set(train.columns)):
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
    expected_cols = pre_cols
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    candidates = [c for c in train.columns if c not in drop]
    feature_cols = []
    for c in candidates:
        s = train[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            if s.nunique(dropna=False) == len(s):
                continue
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique(dropna=False) == len(s):
                continue
        feature_cols.append(c)
    expected_cols = feature_cols
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

X_train = train[expected_cols].copy()

k_range = range(2, 11)
best_k = None
best_score = -np.inf
best_pipeline = None
best_metrics = {}

n = len(X_train)
rng = np.random.RandomState(42)

for k in k_range:
    pipe = Pipeline([
        ('frontend', clone(frontend)),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    fitted = mlca_runtime.fit_record(pipe, X_train, partition='train')
    labels = fitted.predict(X_train)
    X_trans = fitted.named_steps['frontend'].transform(X_train)

    if n > 100000:
        idx = rng.choice(n, size=10000, replace=False)
        X_sub = X_trans[idx]
        labels_sub = labels[idx]
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        sil = silhouette_score(X_sub, labels_sub, random_state=42)
        db = davies_bouldin_score(X_sub, labels_sub)
        ch = calinski_harabasz_score(X_sub, labels_sub)
    else:
        X_dense = X_trans.toarray() if hasattr(X_trans, 'toarray') else X_trans
        sil = silhouette_score(X_dense, labels, random_state=42)
        db = davies_bouldin_score(X_dense, labels)
        ch = calinski_harabasz_score(X_dense, labels)

    aris = []
    for b in range(5):
        weights = rng.multinomial(n, [1.0/n]*n)
        pipe_boot = clone(pipe)
        fitted_boot = mlca_runtime.fit_record(pipe_boot, X_train, partition='train', kmeans__sample_weight=weights)
        labels_boot = fitted_boot.predict(X_train)
        aris.append(adjusted_rand_score(labels, labels_boot))
    stability = float(np.mean(aris))

    if sil > best_score:
        best_score = sil
        best_k = k
        best_pipeline = fitted
        best_metrics = {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch, 'stability': stability}

mlca_runtime.record_metric('silhouette', best_metrics['silhouette'], 'train')
mlca_runtime.record_metric('davies_bouldin', best_metrics['davies_bouldin'], 'train')
mlca_runtime.record_metric('calinski_harabasz', best_metrics['calinski_harabasz'], 'train')
mlca_runtime.record_metric('stability', best_metrics['stability'], 'train')
mlca_runtime.record_metric('n_clusters', best_k, 'train')

mlca_runtime.save_model(best_pipeline, 'model.pkl')

print(f"Chosen k={best_k}")
print(f"Silhouette={best_metrics['silhouette']:.4f}")
print(f"Stability={best_metrics['stability']:.4f}")
print(f"Train rows={n}")
