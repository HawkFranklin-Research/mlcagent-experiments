import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])

def keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return s.dtype.kind in "iu" or s.dtype == object

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not keylike(train[c])]
X = train[feat_cols]

with open(inputs["05_preprocess_fit_preprocessing_FVVN700K/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_FW7XY320/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def contains_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if isinstance(step, ColumnTransformer) or "pre" in name.lower():
            return True
    return False

frontend = sel if contains_pre(sel) else Pipeline([("pre", pre), ("sel", sel)])

mlca_runtime.fit_record(frontend, X, "train")
Xp = frontend.transform(X)

n = len(X)
rng = np.random.RandomState(42)
rows = []
best_k, best_sil = None, -np.inf
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    mlca_runtime.fit_record(km, Xp, "train")
    labels = km.labels_
    sil = float(silhouette_score(Xp, labels, sample_size=min(10000, n), random_state=42))
    db = float(davies_bouldin_score(Xp, labels))
    ch = float(calinski_harabasz_score(Xp, labels))
    aris = []
    for _ in range(5):
        mult = rng.multinomial(n, np.ones(n) / n)
        km_b = KMeans(n_clusters=k, n_init=10, random_state=42)
        mlca_runtime.fit_record(km_b, Xp, "train", sample_weight=mult)
        aris.append(adjusted_rand_score(labels, km_b.labels_))
    rows.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                 "calinski_harabasz": ch, "stability": float(np.mean(aris))})
    if sil > best_sil:
        best_sil, best_k = sil, k

mlca_runtime.save_table(pd.DataFrame(rows).round(4), "cluster_selection.csv",
                        "Cluster count selection")

final = Pipeline([("front", frontend),
                  ("km", KMeans(n_clusters=best_k, n_init=10, random_state=42))])
mlca_runtime.fit_record(final, X, "train")
mlca_runtime.save_model(final, "model.pkl")

mlca_runtime.record_metric("silhouette", best_sil, "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
print(f"rows={n} features={len(feat_cols)} best_k={best_k} silhouette={best_sil:.4f}")
