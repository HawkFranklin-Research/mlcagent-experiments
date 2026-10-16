import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']; excluded = proto['excluded_columns']
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        continue
    feat_cols.append(c)
X = train[feat_cols].copy()

with open(inp['05_preprocess_fit_preprocessing_9D3FJJNM/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_retain_features_Y0S49GVN/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def raw_ok(o):
    fn = getattr(o, 'feature_names_in_', None)
    return fn is not None and list(fn) == list(feat_cols)

def has_step(o, name):
    return isinstance(o, Pipeline) and name in o.named_steps

if raw_ok(sel):
    frontend = sel
elif has_step(pre, 'sel') or has_step(pre, 'selector'):
    frontend = pre
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])

def make_pipe(k):
    return Pipeline([('frontend', frontend),
                     ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=42))])

rows = []
for k in range(2, 11):
    p = make_pipe(k)
    mlca_runtime.fit_record(p, X, 'train')
    lab = p.predict(X)
    Xt = p.named_steps['frontend'].transform(X)
    n = len(Xt)
    sil = silhouette_score(Xt, lab, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(Xt, lab)
    ch = calinski_harabasz_score(Xt, lab)
    rng = np.random.default_rng(42)
    aris = []
    for _ in range(5):
        w = rng.poisson(1.0, size=n).astype(float)
        if w.sum() == 0:
            continue
        pb = make_pipe(k)
        mlca_runtime.fit_record(pb, X, 'train', kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(lab, pb.predict(X)))
    stab = float(np.mean(aris)) if aris else 0.0
    rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db,
                 'calinski_harabasz': ch, 'stability': stab,
                 'score': 0.5 * sil + 0.5 * stab})

res = pd.DataFrame(rows)
best_k = int(res.loc[res['score'].idxmax(), 'k'])
best = make_pipe(best_k)
mlca_runtime.fit_record(best, X, 'train')

mlca_runtime.save_model(best, 'model.pkl')
mlca_runtime.save_table(res.round(4), 'cluster_selection.csv', 'KMeans candidate metrics')
mlca_runtime.record_metric('best_k', float(best_k), 'train')
mlca_runtime.record_metric('best_silhouette',
                           float(res.loc[res['k'] == best_k, 'silhouette'].iloc[0]), 'train')
mlca_runtime.record_metric('best_stability',
                           float(res.loc[res['k'] == best_k, 'stability'].iloc[0]), 'train')
print(f"rows={len(X)} feats={len(feat_cols)} best_k={best_k}")
