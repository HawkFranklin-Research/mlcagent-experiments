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
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing {key}. Available: {list(proto.keys())}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]

def is_keylike(s):
    if s.dtype.kind in "iu" and s.nunique() == len(s):
        return True
    if s.dtype.kind == "O" and s.nunique() == len(s):
        return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not is_keylike(train[c])]
X = train[feat_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_NRDFSV13/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_1934BFX9/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.named_steps[list(obj.named_steps)[0]]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def accepts_raw(obj, cols):
    ic = input_cols(obj)
    return ic is not None and set(ic) == set(cols)

if accepts_raw(sel, feat_cols):
    frontend = sel
elif accepts_raw(pre, feat_cols):
    frontend = Pipeline([("pre", pre), ("sel", sel)])
else:
    frontend = pre

ic = input_cols(frontend)
if ic is not None:
    X = X[ic]

k_range = list(range(2, 9))
rng = np.random.RandomState(42)
results = []
for k in k_range:
    pipe = Pipeline([("front", frontend),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])
    pipe = mlca_runtime.fit_record(pipe, X, "train")
    Xt = pipe[:-1].transform(X)
    labels = pipe.predict(X)
    sil = float(silhouette_score(Xt, labels, sample_size=min(10000, len(X)), random_state=42))
    db = float(davies_bouldin_score(Xt, labels))
    ch = float(calinski_harabasz_score(Xt, labels))
    aris = []
    for b in range(10):
        w = rng.multinomial(len(X), np.ones(len(X)) / len(X))
        pb = Pipeline([("front", frontend),
                       ("km", KMeans(n_clusters=k, n_init=10, random_state=42 + b))])
        pb = mlca_runtime.fit_record(pb, X, "train", km__sample_weight=w)
        aris.append(adjusted_rand_score(labels, pb.predict(X)))
    results.append({"k": k, "silhouette": sil, "db": db, "ch": ch,
                    "stability": float(np.mean(aris))})

res_df = pd.DataFrame(results)
best = res_df.sort_values(["silhouette", "stability"], ascending=False).iloc[0]
best_k = int(best["k"])

final_pipe = Pipeline([("front", frontend),
                       ("km", KMeans(n_clusters=best_k, n_init=10, random_state=42))])
final_pipe = mlca_runtime.fit_record(final_pipe, X, "train")
mlca_runtime.save_model(final_pipe, "model.pkl")

mlca_runtime.record_metric("silhouette", float(best["silhouette"]), "train")
mlca_runtime.record_metric("stability", float(best["stability"]), "train")
mlca_runtime.record_fact_candidate("best_k", best_k, source="model.pkl")
mlca_runtime.save_table(res_df.round(4), "cluster_selection.csv", "KMeans candidate metrics")
print(f"rows={len(X)} features={len(feat_cols)} best_k={best_k} sil={best['silhouette']:.4f}")
