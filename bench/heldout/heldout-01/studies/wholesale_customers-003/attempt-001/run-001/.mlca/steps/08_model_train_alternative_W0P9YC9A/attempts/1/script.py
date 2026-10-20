import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import matplotlib.pyplot as plt

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
drop = set(targets) | set(excluded) | ({group} if group else set())
base_feats = [c for c in train.columns if c not in drop]

def is_key(c):
    s = train[c]
    if s.nunique() != len(train): return False
    if s.dtype == object: return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        u = np.sort(s.unique())
        if len(u) > 1 and u[-1] - u[0] + 1 == len(u):
            return True
    return False

feats_final = []
for c in base_feats:
    if is_key(c): continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part, func in [('year', lambda x: x.dt.year), ('month', lambda x: x.dt.month), ('dayofweek', lambda x: x.dt.dayofweek)]:
            name = f"{c}_{part}"
            train[name] = func(s); val[name] = func(val[c])
            feats_final.append(name)
        continue
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        feats_final.append(c); continue
    if s.nunique() > max(1, int(0.05 * len(train))):
        name = f"{c}_freq"
        vc = train[c].value_counts(normalize=True)
        train[name] = train[c].map(vc).fillna(0.0)
        val[name] = val[c].map(vc).fillna(0.0)
        feats_final.append(name)
    else:
        feats_final.append(c)

numeric = [c for c in feats_final if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
cat = [c for c in feats_final if c not in numeric]

X_train = train[feats_final].copy()

ct = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)
], remainder='drop')
ct.set_output(transform='pandas')
pre = Pipeline([('ct', ct)])
pre.set_output(transform='pandas')
pre = mlca_runtime.fit_record(pre, X_train, partition='train')
Xtr = pre.transform(X_train)

n = len(X_train)
k_range = list(range(2, 9))
res = []
rng = np.random.default_rng(42)
for k in k_range:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=min(1024, n))
    km = mlca_runtime.fit_record(km, Xtr, partition='train')
    labs = km.labels_
    sil = float(silhouette_score(Xtr, labs, sample_size=min(10000, n), random_state=42))
    db = float(davies_bouldin_score(Xtr, labs))
    ch = float(calinski_harabasz_score(Xtr, labs))
    stabs = []
    for _ in range(5):
        w = rng.multinomial(n, np.ones(n) / n)
        km_b = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=1, batch_size=min(1024, n))
        km_b = mlca_runtime.fit_record(km_b, Xtr, partition='train', sample_weight=w)
        stabs.append(float(adjusted_rand_score(labs, km_b.labels_)))
    stab = float(np.mean(stabs))
    res.append(dict(k=k, silhouette=sil, davies_bouldin=db, calinski_harabasz=ch, stability=stab))

best = max(res, key=lambda r: (r['silhouette'], r['stability']))
final_k = best['k']

final_ct = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)
], remainder='drop')
final_ct.set_output(transform='pandas')
final_pipe = Pipeline([
    ('ct', final_ct),
    ('km', MiniBatchKMeans(n_clusters=final_k, random_state=42, n_init=10, batch_size=min(1024, n)))
])
final_pipe.set_output(transform='pandas')
final_pipe = mlca_runtime.fit_record(final_pipe, X_train, partition='train')
mlca_runtime.save_model(final_pipe, "alternative_model.pkl")

for r in res:
    if r['k'] == final_k:
        mlca_runtime.record_metric('silhouette', r['silhouette'], split='train')
        mlca_runtime.record_metric('davies_bouldin', r['davies_bouldin'], split='train')
        mlca_runtime.record_metric('calinski_harabasz', r['calinski_harabasz'], split='train')
        mlca_runtime.record_metric('stability_ari', r['stability'], split='train')
        mlca_runtime.record_metric('n_clusters', float(final_k), split='train')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([r['k'] for r in res], [r['silhouette'] for r in res], marker='o')
ax.set_xlabel('k'); ax.set_ylabel('silhouette'); ax.set_title('Silhouette by cluster count')
mlca_runtime.save_figure(fig, 'silhouette_by_k.png')

print(f"rows={n}, features={len(feats_final)}, best_k={final_k}")
print(f"silhouette={best['silhouette']:.4f} stability={best['stability']:.4f}")
