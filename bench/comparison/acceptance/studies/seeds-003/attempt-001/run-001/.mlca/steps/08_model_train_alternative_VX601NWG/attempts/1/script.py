import mlca_runtime
import json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
primary_metric = protocol["task"]["primary_metric"]

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
for c in list(feature_cols):
    if train[c].dtype == object and train[c].nunique() == len(train):
        feature_cols.remove(c)

X = train[feature_cols].copy()
n = len(X)
ks = list(range(2, min(8, n - 1) + 1))

def make_pipe(k):
    pre = ColumnTransformer([('num', StandardScaler(), feature_cols)], remainder='drop')
    pre.set_output(transform='pandas')
    return Pipeline([('pre', pre), ('km', MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10))])

results = []
rng = np.random.default_rng(42)
B = 20
for k in ks:
    pipe = make_pipe(k)
    fitted = mlca_runtime.fit_record(pipe, X, 'train')
    labels = fitted.named_steps['km'].labels_
    Xt = fitted.named_steps['pre'].transform(X)
    sil = silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(Xt, labels)
    ch = calinski_harabasz_score(Xt, labels)
    aris = []
    for b in range(B):
        w = rng.multinomial(n, np.ones(n) / n).astype(float)
        p2 = make_pipe(k)
        f2 = mlca_runtime.fit_record(p2, X, 'train', km__sample_weight=w)
        aris.append(adjusted_rand_score(labels, f2.named_steps['km'].labels_))
    stab = float(np.mean(aris))
    results.append(dict(k=k, silhouette=sil, davies_bouldin=db, calinski_harabasz=ch,
                        stability=stab, pipe=fitted))

best_sil = max(r['silhouette'] for r in results)
cand = [r for r in results if r['silhouette'] >= best_sil - 0.005]
best = max(cand, key=lambda r: r['stability'])
mlca_runtime.save_model(best['pipe'], "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", float(best['silhouette']), "train")
mlca_runtime.record_metric("stability_ari", float(best['stability']), "train")
mlca_runtime.record_metric("davies_bouldin", float(best['davies_bouldin']), "train")
mlca_runtime.record_metric("calinski_harabasz", float(best['calinski_harabasz']), "train")
print(f"selected k={best['k']} silhouette={best['silhouette']:.4f} stability={best['stability']:.3f}")
