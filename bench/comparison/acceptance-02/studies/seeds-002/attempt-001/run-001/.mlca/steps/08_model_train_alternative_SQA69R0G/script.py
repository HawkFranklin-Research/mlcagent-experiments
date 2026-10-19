import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(proto)}")

targets = set(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto["group_column"]
pm = proto["task"].get("primary_metric") or "silhouette"

train = pd.read_parquet(inp["train"])
n = len(train)

def key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == n
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c])
]
num_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_datetime64_any_dtype(train[c])
]
cat_cols = [
    c for c in feature_cols
    if c not in num_cols
    and not pd.api.types.is_datetime64_any_dtype(train[c])
    and train[c].nunique(dropna=False) <= max(5, int(0.05 * n))
]

transformers = []
if num_cols:
    transformers.append((
        "num",
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
        num_cols,
    ))
if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]),
        cat_cols,
    ))

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")
X = train[feature_cols].copy()

max_k = min(10, max(2, n // 20))
k_range = range(2, max_k + 1)
rng = np.random.RandomState(42)
B = 5
best = None

for k in k_range:
    pipe = Pipeline([
        ("pre", clone(pre)),
        ("cluster", MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=10, batch_size=min(1024, max(2, n))
        )),
    ])
    mlca_runtime.fit_record(pipe, X, "train")
    labels = pipe.predict(X)
    Xt = pipe.named_steps["pre"].transform(X)

    sil = silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(Xt, labels)
    ch = calinski_harabasz_score(Xt, labels)

    aris = []
    for _ in range(B):
        w = rng.multinomial(n, np.ones(n) / n)
        pb = clone(pipe)
        mlca_runtime.fit_record(pb, X, "train", cluster__sample_weight=w)
        aris.append(adjusted_rand_score(labels, pb.predict(X)))
    stab = float(np.mean(aris))

    score = sil + 0.25 * stab
    rec = {"k": k, "pipe": pipe, "sil": sil, "db": db, "ch": ch, "stab": stab, "score": score}
    if best is None or rec["score"] > best["score"]:
        best = rec

mlca_runtime.save_model(best["pipe"], "alternative_model.pkl")
mlca_runtime.record_metric("selected_k", float(best["k"]), "train")
mlca_runtime.record_metric(pm, float(best["sil"]), "train")
mlca_runtime.record_metric("stability_ari", float(best["stab"]), "train")
mlca_runtime.record_metric("davies_bouldin", float(best["db"]), "train")
mlca_runtime.record_metric("calinski_harabasz", float(best["ch"]), "train")
print(f"rows={n} features={len(feature_cols)} k={best['k']} silhouette={best['sil']:.4f} stability={best['stab']:.4f}")
