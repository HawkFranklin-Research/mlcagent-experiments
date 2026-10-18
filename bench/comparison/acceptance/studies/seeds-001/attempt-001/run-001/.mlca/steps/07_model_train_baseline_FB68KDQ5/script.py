import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise ValueError(f"Missing protocol key {key!r}; available keys: {list(proto)}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

for path in (
    "05_preprocess_fit_preprocessing_A5A2YZJ3/preprocessor.pkl",
    "06_features_retain_features_297WVMMN/selector_all.pkl",
):
    pickle.loads(inp[path].read_bytes())
for path in (
    "05_preprocess_fit_preprocessing_A5A2YZJ3/manifest.json",
    "06_features_retain_features_297WVMMN/manifest.json",
):
    json.loads(inp[path].read_text())


def is_keylike(s: pd.Series) -> bool:
    n = len(s)
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    return False


feature_cols = [
    c
    for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_keylike(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions")

X = train[feature_cols].copy()
non_numeric = [
    c
    for c in feature_cols
    if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
]
if non_numeric:
    raise ValueError(f"Expected numeric seed features; non-numeric columns: {non_numeric}")

n = len(X)
k_max = min(10, n - 1)
if k_max < 2:
    raise ValueError("Need at least three training rows for clustering")

rng = np.random.default_rng(42)
results = []

for k in range(2, k_max + 1):
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
        ]
    )
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.named_steps["kmeans"].labels_
    Xs = fitted.named_steps["scale"].transform(X)
    Xs_df = pd.DataFrame(Xs, index=X.index, columns=feature_cols)

    sil = silhouette_score(
        Xs, labels, sample_size=min(10000, n), random_state=42
    )
    db = davies_bouldin_score(Xs, labels)
    ch = calinski_harabasz_score(Xs, labels)

    aris = []
    for b in range(5):
        weights = rng.multinomial(n, np.full(n, 1.0 / n))
        km = KMeans(n_clusters=k, random_state=100 + b, n_init=10)
        mlca_runtime.fit_record(km, Xs_df, "train", sample_weight=weights)
        aris.append(adjusted_rand_score(labels, km.labels_))

    results.append(
        {
            "k": k,
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "stability": float(np.mean(aris)),
        }
    )


def norm(values, higher_is_better=True):
    a = np.asarray(values, dtype=float)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.full_like(a, 0.5)
    z = (a - lo) / (hi - lo)
    return z if higher_is_better else 1.0 - z


sil_s = norm([r["silhouette"] for r in results], True)
db_s = norm([r["davies_bouldin"] for r in results], False)
ch_s = norm([r["calinski_harabasz"] for r in results], True)
st_s = norm([r["stability"] for r in results], True)
best_i = int(np.argmax((sil_s + db_s + ch_s + st_s) / 4.0))
best = results[best_i]
best_k = best["k"]

final_pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10)),
    ]
)
final_fit = mlca_runtime.fit_record(final_pipe, X, "train")
mlca_runtime.save_model(final_fit, "model.pkl")

mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("davies_bouldin", best["davies_bouldin"], "train")
mlca_runtime.record_metric("calinski_harabasz", best["calinski_harabasz"], "train")
mlca_runtime.record_metric("stability_ari", best["stability"], "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")

print(f"rows={n} features={len(feature_cols)}")
print(f"selected k={best_k}")
print(f"silhouette={best['silhouette']:.4f} stability={best['stability']:.4f}")
