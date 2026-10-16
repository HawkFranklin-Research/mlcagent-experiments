import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
all_cols = list(train.columns)
feature_cols = []
for c in all_cols:
    if c in targets or c in excluded or c == group_col:
        continue
    if train[c].dtype == object or pd.api.types.is_integer_dtype(train[c].dtype):
        if train[c].nunique(dropna=False) == len(train):
            continue
    feature_cols.append(c)

X_train = train[feature_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_RHW0T9SM/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

if isinstance(sel, Pipeline) and len(sel.steps) > 1:
    frontend = sel
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])

def make_pipeline(k):
    return Pipeline([
        ("frontend", frontend),
        ("cluster", MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3))
    ])

k_range = range(2, 11)
results = []
for k in k_range:
    pipe = make_pipeline(k)
    pipe = mlca_runtime.fit_record(pipe, X_train, "train")
    X_trans = pipe[:-1].transform(X_train)
    labels = pipe.named_steps["cluster"].labels_
    n = len(X_train)
    if n > 10000:
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        X_metric = X_trans[idx]
        labels_metric = labels[idx]
    else:
        X_metric = X_trans
        labels_metric = labels
    if hasattr(X_metric, "toarray"):
        X_metric = X_metric.toarray()
    sil = silhouette_score(X_metric, labels_metric, sample_size=min(10000, len(X_metric)), random_state=42)
    db = davies_bouldin_score(X_metric, labels_metric)
    ch = calinski_harabasz_score(X_metric, labels_metric)
    aris = []
    for b in range(3):
        w = np.random.poisson(1, size=n)
        pipe_b = make_pipeline(k)
        pipe_b = mlca_runtime.fit_record(pipe_b, X_train, "train", **{"cluster__sample_weight": w})
        labels_b = pipe_b.named_steps["cluster"].labels_
        aris.append(adjusted_rand_score(labels, labels_b))
    stab = float(np.mean(aris))
    results.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "stability": stab})
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

best = max(results, key=lambda r: (r["silhouette"], r["stability"]))
best_k = best["k"]
mlca_runtime.record_metric("selected_k", best_k, "train")
mlca_runtime.record_metric("selected_silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("selected_stability", best["stability"], "train")

final_pipe = make_pipeline(best_k)
final_pipe = mlca_runtime.fit_record(final_pipe, X_train, "train")
mlca_runtime.save_model(final_pipe, "alternative_model.pkl")

print(f"Train shape: {X_train.shape}")
print(f"Selected k: {best_k}")
print(f"Silhouette: {best['silhouette']:.4f}")
print(f"Stability: {best['stability']:.4f}")
