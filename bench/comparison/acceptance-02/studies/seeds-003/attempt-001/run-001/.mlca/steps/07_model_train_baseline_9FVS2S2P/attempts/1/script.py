import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from scipy import sparse

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_Y1MSMN58/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    X = train[list(pre.feature_names_in_)].copy()
else:
    drop = set(protocol.get("targets", [])) | set(protocol.get("excluded_columns", []))
    if protocol.get("group_column"):
        drop.add(protocol["group_column"])
    n = len(train)
    keep = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if s.nunique() < n:
                keep.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == n:
            continue
        else:
            keep.append(c)
    X = train[keep].copy()
print("X", X.shape)

def make_pipe(k):
    return Pipeline([("pre", pre),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])

cands = []
for k in range(2, 9):
    fitted = mlca_runtime.fit_record(make_pipe(k), X, "train")
    Xt = fitted["pre"].transform(X)
    if hasattr(Xt, "values"):
        Xt = Xt.values
    if sparse.issparse(Xt):
        Xt = Xt.toarray()
    lab = fitted["km"].labels_
    cands.append({
        "k": k, "pipe": fitted, "Xt": Xt,
        "sil": float(silhouette_score(Xt, lab,
                                      sample_size=min(10000, len(Xt)),
                                      random_state=42)),
        "db": float(davies_bouldin_score(Xt, lab)),
        "ch": float(calinski_harabasz_score(Xt, lab)),
    })

rng = np.random.default_rng(42)
for c in cands:
    Xt = c["Xt"]
    base = c["pipe"]["km"].labels_
    aris = []
    for _ in range(5):
        w = rng.multinomial(len(Xt), np.ones(len(Xt)) / len(Xt)).astype(float)
        km = KMeans(n_clusters=c["k"], n_init=5, random_state=42)
        mlca_runtime.fit_record(km, Xt, "train", sample_weight=w)
        aris.append(adjusted_rand_score(base, km.labels_))
    c["stab"] = float(np.mean(aris))

best = max(cands, key=lambda c: c["sil"] + 0.3 * c["stab"])
print(f"best k={best['k']} sil={best['sil']:.3f} stab={best['stab']:.3f}")

mlca_runtime.save_model(best["pipe"], "model.pkl")
mlca_runtime.record_metric("best_k", float(best["k"]), "train")
mlca_runtime.record_metric("silhouette", best["sil"], "train")
mlca_runtime.record_metric("davies_bouldin", best["db"], "train")
mlca_runtime.record_metric("calinski_harabasz", best["ch"], "train")
mlca_runtime.record_metric("stability_ari", best["stab"], "train")
