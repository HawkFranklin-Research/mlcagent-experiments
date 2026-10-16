import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = list(proto.get("targets") or [])
excluded = list(proto.get("excluded_columns") or [])
group = proto.get("group_column")
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("string"):
        if s.nunique(dropna=True) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

X = train[feature_cols].copy()
Xv = val[feature_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_B48XD0NW/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_S94R336H/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def step_names(o):
    return list(o.named_steps.keys()) if isinstance(o, Pipeline) else []

pre_names = step_names(pre)
sel_names = step_names(sel)
if isinstance(sel, Pipeline) and any(n in sel_names for n in pre_names):
    base = sel
else:
    base = Pipeline([("pre", pre), ("sel", sel)])

n = len(X)
rng = np.random.default_rng(42)
B = 10
results = []
for k in range(2, 11):
    pipe = Pipeline([("base", base), ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    Xb = fitted.named_steps["base"].transform(X)
    if hasattr(Xb, "toarray"):
        Xb = Xb.toarray()
    labels = fitted.named_steps["km"].labels_
    sil = float(silhouette_score(Xb, labels, sample_size=min(10000, n), random_state=42))
    db = float(davies_bouldin_score(Xb, labels))
    ch = float(calinski_harabasz_score(Xb, labels))
    aris = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        km = mlca_runtime.fit_record(
            KMeans(n_clusters=k, n_init=10, random_state=42), Xb, "train", sample_weight=w)
        aris.append(adjusted_rand_score(labels, km.labels_))
    stab = float(np.mean(aris))
    results.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "stability": stab,
                    "score": sil + stab})

best = max(results, key=lambda r: r["score"])
best_k = best["k"]

final = Pipeline([("base", base), ("km", KMeans(n_clusters=best_k, n_init=10, random_state=42))])
final = mlca_runtime.fit_record(final, X, "train")
mlca_runtime.save_model(final, "model.pkl")

tr_labels = final.named_steps["km"].labels_
va_labels = final.named_steps["km"].predict(final.named_steps["base"].transform(Xv))
mlca_runtime.save_row_outputs(pd.DataFrame({"row_id": train.index, "cluster": tr_labels}), "train_clusters")
mlca_runtime.save_row_outputs(pd.DataFrame({"row_id": val.index, "cluster": va_labels}), "val_clusters")

mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("stability", best["stability"], "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
mlca_runtime.save_table(pd.DataFrame(results), "cluster_selection", "KMeans candidate metrics")

print(f"features={len(feature_cols)} train={n} val={len(Xv)}")
print(f"best_k={best_k} silhouette={best['silhouette']:.4f} stability={best['stability']:.4f}")
