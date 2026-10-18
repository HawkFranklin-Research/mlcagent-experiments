import mlca_runtime
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
    primary_metric = task["primary_metric"]
except KeyError as e:
    raise KeyError(
        f"Missing protocol key {e}. Available top-level keys: {list(protocol.keys())}"
    )

group = protocol.get("group_column", None)
train = pd.read_parquet(inp["train"])

key_like = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_string_dtype(train[c]) or train[c].dtype == object:
        if train[c].nunique(dropna=False) == len(train[c]):
            key_like.append(c)

feature_cols = [
    c
    for c in train.columns
    if c not in targets and c not in excluded and c != group and c not in key_like
]
if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions.")
X = train[feature_cols].copy()
n = len(X)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accepts_raw(obj, cols):
    if obj is None:
        return False
    if isinstance(obj, Pipeline):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names) == list(cols)
        if obj.steps:
            first = obj.steps[0][1]
            names = getattr(first, "feature_names_in_", None)
            if names is not None:
                return list(names) == list(cols)
        nf = getattr(obj, "n_features_in_", None)
        return nf == len(cols) and len(obj.steps) >= 2
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_) == list(cols)
    nf = getattr(obj, "n_features_in_", None)
    return nf == len(cols)


pre = load_pkl(inp["05_preprocess_fit_preprocessing_A5A2YZJ3/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_297WVMMN/selector_all.pkl"])

frontend = None
if isinstance(sel, Pipeline) and accepts_raw(sel, feature_cols):
    frontend = sel
elif accepts_raw(pre, feature_cols):
    frontend = pre
else:
    frontend = Pipeline([("scaler", StandardScaler())])
    try:
        frontend.set_output(transform="pandas")
    except Exception:
        pass

k_values = [2, 3, 4, 5, 6]
rng = np.random.RandomState(42)
results = []

for k in k_values:
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=256,
        n_init=10,
        max_iter=100,
    )
    pipe = Pipeline([("frontend", clone(frontend)), ("kmeans", km)])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.named_steps["kmeans"].labels_

    Xt = fitted.named_steps["frontend"].transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    elif isinstance(Xt, pd.DataFrame):
        Xt = Xt.to_numpy()

    sil = float(
        silhouette_score(
            Xt,
            labels,
            sample_size=min(10000, n) if n > 10000 else None,
            random_state=42,
        )
    )
    db = float(davies_bouldin_score(Xt, labels))
    ch = float(calinski_harabasz_score(Xt, labels))

    aris = []
    for b in range(5):
        w = rng.poisson(1.0, size=n).astype(float)
        if w.sum() == 0:
            w[:] = 1.0
        kmb = MiniBatchKMeans(
            n_clusters=k,
            random_state=42 + b,
            batch_size=256,
            n_init=10,
            max_iter=100,
        )
        pipeb = Pipeline([("frontend", clone(frontend)), ("kmeans", kmb)])
        fitb = mlca_runtime.fit_record(
            pipeb, X, "train", kmeans__sample_weight=w
        )
        lb = fitb.named_steps["kmeans"].labels_
        aris.append(float(adjusted_rand_score(labels, lb)))

    stab = float(np.mean(aris)) if aris else 0.0
    results.append(dict(k=k, sil=sil, db=db, ch=ch, stab=stab))

    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"davies_bouldin_k{k}", db, "train")
    mlca_runtime.record_metric(f"calinski_harabasz_k{k}", ch, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

sils = np.array([r["sil"] for r in results], float)
dbs = np.array([r["db"] for r in results], float)
chs = np.array([r["ch"] for r in results], float)
stabs = np.array([r["stab"] for r in results], float)


def norm(v, higher=True):
    if v.max() == v.min():
        return np.ones_like(v)
    if higher:
        return (v - v.min()) / (v.max() - v.min())
    return (v.max() - v) / (v.max() - v.min())


score = (
    0.5 * norm(sils, True)
    + 0.2 * norm(stabs, True)
    + 0.15 * norm(chs, True)
    + 0.15 * norm(dbs, False)
)
best = results[int(np.argmax(score))]
best_k = best["k"]

km = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    batch_size=256,
    n_init=10,
    max_iter=100,
)
final_pipe = Pipeline([("frontend", clone(frontend)), ("kmeans", km)])
final_pipe = mlca_runtime.fit_record(final_pipe, X, "train")
mlca_runtime.save_model(final_pipe, "alternative_model.pkl")

mlca_runtime.record_metric("selected_k", float(best_k), "train")
mlca_runtime.record_metric("selected_silhouette", float(best["sil"]), "train")
mlca_runtime.record_metric("selected_stability", float(best["stab"]), "train")
mlca_runtime.record_metric("selected_db", float(best["db"]), "train")
mlca_runtime.record_metric("selected_ch", float(best["ch"]), "train")

print(f"features={len(feature_cols)}")
print(f"k={best_k}")
print(f"silhouette={best['sil']:.4f}")
print(f"stability={best['stab']:.4f}")
print("saved=alternative_model.pkl")
