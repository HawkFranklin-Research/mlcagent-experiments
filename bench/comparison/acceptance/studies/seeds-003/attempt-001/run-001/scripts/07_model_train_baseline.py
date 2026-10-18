import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {list(proto)}")
targets = set(proto["targets"])
excluded = set(proto["excluded_columns"])
group_col = proto.get("group_column")
task = proto["task"]
primary_metric = task.get("primary_metric") or "silhouette"

train = pd.read_parquet(inputs["train"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s) and s.dropna().is_monotonic_increasing
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if key_like(train[c]):
        continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions.")

with open(inputs["06_features_retain_features_JQHA8ERZ/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["05_preprocess_fit_preprocessing_0JVWWBDZ/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

def has_preproc(obj):
    return isinstance(obj, Pipeline) and any("preproc" in n.lower() or n.lower() == "prep" for n in obj.named_steps)

def raw_ok(obj):
    return hasattr(obj, "feature_names_in_") and set(map(str, obj.feature_names_in_)) == set(map(str, feature_cols))

if isinstance(selector, Pipeline) and has_preproc(selector):
    base = selector
elif raw_ok(selector):
    base = selector
elif raw_ok(preprocessor):
    base = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
elif isinstance(preprocessor, Pipeline) and has_preproc(preprocessor):
    base = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
else:
    raise ValueError(
        f"Cannot build a raw-feature frontend from saved objects; "
        f"selector={type(selector)}, preprocessor={type(preprocessor)}, "
        f"available keys={list(proto)}"
    )

if hasattr(base, "feature_names_in_"):
    expected = list(base.feature_names_in_)
    if set(map(str, expected)) != set(map(str, feature_cols)):
        raise ValueError("Saved frontend expects a different raw feature column set.")
    feature_cols = expected

X = train[feature_cols].copy()
n = len(X)
max_k = min(8, n - 1)
if max_k < 2:
    raise ValueError(f"Need at least 3 rows for clustering; got {n}.")
k_values = list(range(2, max_k + 1))
B = 10
records = []

for k in k_values:
    pipe = Pipeline([
        ("frontend", clone(base)),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
    ])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.named_steps["kmeans"].labels_
    Xt = fitted.named_steps["frontend"].transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = pd.DataFrame(Xt, index=X.index) if not isinstance(Xt, pd.DataFrame) else Xt
    sample_size = min(10000, len(Xt))
    sil = float(silhouette_score(Xt, labels, sample_size=sample_size, random_state=42))
    db = float(davies_bouldin_score(Xt, labels))
    ch = float(calinski_harabasz_score(Xt, labels))
    aris = []
    for b in range(B):
        rng = np.random.default_rng(42 + 1000 * k + b)
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        km = KMeans(n_clusters=k, random_state=42 + b, n_init=10)
        km_fit = mlca_runtime.fit_record(km, Xt, "train", sample_weight=w)
        aris.append(adjusted_rand_score(labels, km_fit.labels_))
    records.append({
        "k": k,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "stability": float(np.mean(aris)),
        "pipe": fitted,
    })

best = max(records, key=lambda r: (r["silhouette"], r["stability"], -r["k"]))
mlca_runtime.save_model(best["pipe"], "model.pkl")
mlca_runtime.record_metric("n_clusters", float(best["k"]), "train")
mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("davies_bouldin", best["davies_bouldin"], "train")
mlca_runtime.record_metric("calinski_harabasz", best["calinski_harabasz"], "train")
mlca_runtime.record_metric("stability_ari", best["stability"], "train")

print(f"features={len(feature_cols)} train_rows={n}")
print(f"candidate_k={k_values}")
print(f"selected_k={best['k']} silhouette={best['silhouette']:.4f} stability={best['stability']:.4f}")
