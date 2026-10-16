import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_YJWZ54M0/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]
X = train[feat_cols]

def includes_preprocessing(obj):
    if isinstance(obj, Pipeline):
        first_name = list(obj.named_steps.keys())[0]
        first_step = obj.named_steps[first_name]
        return 'ColumnTransformer' in type(first_step).__name__ or 'pre' in first_name.lower()
    return False

frontend = sel if includes_preprocessing(sel) else Pipeline([('pre', pre), ('sel', sel)])

ks = [2, 3, 4, 5, 6, 7, 8]
results = []
for k in ks:
    pipe = Pipeline([('front', frontend), ('km', KMeans(n_clusters=k, n_init=10, random_state=42))])
    fitted = mlca_runtime.fit_record(pipe, X, 'train')
    labels = fitted.predict(X)
    Xt = fitted[:-1].transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    sil = silhouette_score(Xt, labels, sample_size=min(10000, len(Xt)), random_state=42)
    db = davies_bouldin_score(Xt, labels)
    ch = calinski_harabasz_score(Xt, labels)
    rng = np.random.RandomState(42)
    stab = []
    for b in range(5):
        w = rng.multinomial(len(X), np.ones(len(X)) / len(X))
        pipe_b = Pipeline([('front', frontend), ('km', KMeans(n_clusters=k, n_init=10, random_state=42))])
        fb = mlca_runtime.fit_record(pipe_b, X, 'train', km__sample_weight=w)
        lb = fb.predict(X)
        cm = confusion_matrix(labels, lb)
        row, col = linear_sum_assignment(-cm)
        stab.append(cm[row, col].sum() / len(labels))
    stab_mean = float(np.mean(stab))
    results.append({'k': k, 'silhouette': sil, 'davies_bouldin': db,
                    'calinski_harabasz': ch, 'stability': stab_mean})
    mlca_runtime.record_metric(f'silhouette_k{k}', sil, 'train')
    mlca_runtime.record_metric(f'stability_k{k}', stab_mean, 'train')

best = max(results, key=lambda r: r['silhouette'] + r['stability'])
k_best = best['k']
mlca_runtime.record_metric('selected_k', k_best, 'train')

final = Pipeline([('front', frontend), ('km', KMeans(n_clusters=k_best, n_init=10, random_state=42))])
mlca_runtime.fit_record(final, X, 'train')
mlca_runtime.save_model(final, 'model.pkl')

print(f"Selected k={k_best}, silhouette={best['silhouette']:.4f}, stability={best['stability']:.4f}")
