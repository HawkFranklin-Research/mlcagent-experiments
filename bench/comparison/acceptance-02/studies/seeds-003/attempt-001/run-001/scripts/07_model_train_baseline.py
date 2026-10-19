import mlca_runtime
import json, pickle, copy
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
    feat_cols = list(pre.feature_names_in_)
else:
    drop = set(protocol.get("targets", [])) | set(protocol.get("excluded_columns", []))
    if protocol.get("group_column"):
        drop.add(protocol["group_column"])
    feat_cols = [c for c in train.columns if c not in drop]
X = train[feat_cols].copy()

try:
    pre.set_output(transform="pandas")
except Exception:
    pass

def to_arr(Xt):
    if sparse.issparse(Xt):
        return Xt.toarray()
    return np.asarray(Xt)

def make_pipe(k):
    return Pipeline([("pre", copy.deepcopy(pre)),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])

cands = []
for k in range(2, 9):
    fitted = mlca_runtime.fit_record(make_pipe(k), X, "train")
    Xt = to_arr(fitted.named_steps["pre"].transform(X))
    lab = np.asarray(fitted.named_steps["km"].labels_)
    cands.append({
        "k": k, "pipe": fitted, "Xt": Xt, "lab": lab,
        "sil": float(silhouette_score(Xt, lab,
                                      sample_size=min(10000, len(Xt)),
                                      random_state=42)),
        "db": float(davies_bouldin_score(Xt, lab)),
        "ch": float(calinski_harabasz_score(Xt, lab)),
    })

rng = np.random.default_rng(42)
n = len(X)
for c in cands:
    aris = []
    base = c["lab"]
    for _ in range(5):
        w = rng.multinomial(n, np.ones(n) / n).astype(float)
        pipe_b = make_pipe(c["k"])
        fitted_b = mlca_runtime.fit_record(pipe_b, X, "train", km__sample_weight=w)
        aris.append(adjusted_rand_score(base, fitted_b.named_steps["km"].labels_))
    c["stab"] = float(np.mean(aris))

best = max(cands, key=lambda c: c["sil"] + 0.3 * c["stab"])
print(f"X={X.shape} k_best={best['k']} sil={best['sil']:.3f} stab={best['stab']:.3f}")

mlca_runtime.save_model(best["pipe"], "model.pkl")
mlca_runtime.record_metric("best_k", float(best["k"]), "train")
mlca_runtime.record_metric("silhouette", best["sil"], "train")
mlca_runtime.record_metric("davies_bouldin", best["db"], "train")
mlca_runtime.record_metric("calinski_harabasz", best["ch"], "train")
mlca_runtime.record_metric("stability_ari", best["stab"], "train")
