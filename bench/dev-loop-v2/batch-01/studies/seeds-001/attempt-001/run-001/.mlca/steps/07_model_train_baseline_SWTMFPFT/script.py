import json
import numpy as np
import pandas as pd
import mlca_runtime as m
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score)

m.set_seed(42)
inp = m.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns', []))
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
n = len(train)
bad = set(targets) | set(excluded) | ({group} if group else set())
cols = [c for c in train.columns if c not in bad]
feat_cols = []
for c in cols:
    s = train[c]
    if s.nunique() == n and (s.dtype == object or s.dtype.kind in 'iu'):
        continue
    feat_cols.append(c)

num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.nunique() <= max(2, int(0.05 * n)):
        cat_cols.append(c)

schema = num_cols + cat_cols
X = train[schema].copy()

def make_pre():
    ts = []
    if num_cols:
        ts.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        ts.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    return ColumnTransformer(ts, remainder='drop').set_output(transform='pandas')

pre = m.fit_record(make_pre(), X, 'train')
Xt = pre.transform(X)

candidates = []
for k in range(2, 9):
    km = m.fit_record(KMeans(n_clusters=k, n_init=20, random_state=42), Xt, 'train')
    lab = km.labels_
    sil = silhouette_score(Xt, lab, sample_size=min(10000, len(Xt)), random_state=42)
    db = davies_bouldin_score(Xt, lab)
    ch = calinski_harabasz_score(Xt, lab)
    aris = []
    for b in range(10):
        rng = np.random.RandomState(1000 + k * 100 + b)
        w = rng.multinomial(len(Xt), [1.0 / len(Xt)] * len(Xt)).astype(float)
        kmb = m.fit_record(KMeans(n_clusters=k, n_init=10, random_state=42 + k * 10 + b),
                           Xt, 'train', sample_weight=w)
        aris.append(adjusted_rand_score(lab, kmb.labels_))
    ari = float(np.mean(aris))
    candidates.append({'k': k, 'silhouette': float(sil), 'stability_ari': ari,
                       'davies_bouldin': float(db), 'calinski_harabasz': float(ch)})

elig = [r for r in candidates if r['stability_ari'] >= 0.6]
best = max(elig if elig else candidates, key=lambda r: r['silhouette'])
best_k = best['k']

final = Pipeline([('pre', make_pre()),
                  ('km', KMeans(n_clusters=best_k, n_init=50, random_state=42))])
final = m.fit_record(final, X, 'train')
m.save_model(final, 'model.pkl')

m.record_metric('silhouette', best['silhouette'], 'train')
m.record_metric('stability_ari', best['stability_ari'], 'train')
m.record_metric('selected_k', float(best_k), 'train')
m.record_metric('davies_bouldin', best['davies_bouldin'], 'train')
m.record_metric('calinski_harabasz', best['calinski_harabasz'], 'train')
m.save_table(pd.DataFrame(candidates).round(4), 'cluster_candidates.csv',
             'KMeans candidates on full training rows')

print(f"n={n} features={len(schema)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"best_k={best_k} silhouette={best['silhouette']:.4f} stability_ari={best['stability_ari']:.3f}")
