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
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_RHW0T9SM/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for s in obj.named_steps.values():
            c = input_cols(s)
            if c:
                return c
    return None

def contains(outer, inner):
    if not isinstance(outer, Pipeline):
        return False
    for s in outer.named_steps.values():
        if s is inner or contains(s, inner):
            return True
    return False

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)

if contains(pre, sel):
    frontend = pre
elif contains(sel, pre):
    frontend = sel
elif sel_cols is not None and pre_cols is not None and set(sel_cols) == set(pre_cols):
    frontend = sel
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

raw_cols = input_cols(frontend) or pre_cols
if raw_cols is None:
    raw_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]
X_train = train[raw_cols].copy()

def to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)

results = []
for k in range(2, 11):
    pipe = Pipeline([("frontend", frontend),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42))])
    pipe = mlca_runtime.fit_record(pipe, X_train, "train")
    Xt = to_dense(pipe.named_steps["frontend"].transform(X_train))
    labels = pipe.named_steps["kmeans"].labels_
    n = len(Xt)
    sil = silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(Xt, labels)
    ch = calinski_harabasz_score(Xt, labels)
    rng = np.random.RandomState(42)
    Xt_df = pd.DataFrame(Xt, index=X_train.index)
    aris = []
    for _ in range(5):
        w = rng.multinomial(n, np.ones(n) / n).astype(float)
        km = mlca_runtime.fit_record(
            KMeans(n_clusters=k, n_init=10, random_state=42),
            Xt_df, "train", sample_weight=w)
        aris.append(adjusted_rand_score(labels, km.labels_))
    results.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "stability": float(np.mean(aris))})

res_df = pd.DataFrame(results)
res_df["score"] = res_df["silhouette"] + res_df["stability"]
best = res_df.loc[res_df["score"].idxmax()]
best_k = int(best["k"])

mlca_runtime.save_table(res_df.round(4), "cluster_selection.csv",
                        "KMeans candidate metrics on training rows")
mlca_runtime.record_metric("silhouette", float(best["silhouette"]), "train")
mlca_runtime.record_metric("stability", float(best["stability"]), "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")

final = Pipeline([("frontend", frontend),
                  ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=42))])
final = mlca_runtime.fit_record(final, X_train, "train")
mlca_runtime.save_model(final, "model.pkl")

print(f"train rows={len(train)} features={len(raw_cols)}")
print(f"best k={best_k} silhouette={best['silhouette']:.4f} stability={best['stability']:.4f}")
