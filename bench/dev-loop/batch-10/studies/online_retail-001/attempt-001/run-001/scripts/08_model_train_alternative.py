import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available: {list(protocol)}")
targets = list(protocol.get("targets") or [])
excluded = set(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
n = len(train)
if n < 3:
    raise ValueError("Need at least 3 training rows for clustering")

with open(inputs["05_preprocess_fit_preprocessing_KQAA3HGR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) > max(10, 0.05 * n):
            continue
    if pd.api.types.is_integer_dtype(s):
        vals = set(s.dropna().unique())
        if s.nunique(dropna=True) == n and n > 1 and not vals.issubset({0, 1}):
            continue
    feature_cols.append(c)

if hasattr(pre, "feature_names_in_"):
    input_cols = [c for c in pre.feature_names_in_ if c in train.columns and
                  c not in targets and c not in excluded and c != group_col]
else:
    input_cols = feature_cols
if not input_cols:
    raise ValueError("No usable feature columns")
X = train[input_cols]

def make_pipe(k):
    return Pipeline([
        ("preprocess", clone(pre)),
        ("kmeans", MiniBatchKMeans(n_clusters=k, random_state=42,
                                    n_init=10, batch_size=1024, max_iter=100)),
    ])

max_k = min(10, n - 1)
k_values = list(range(2, max_k + 1))
rng = np.random.default_rng(42)
results = []

for k in k_values:
    pipe = make_pipe(k)
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.predict(X)
    idx = rng.choice(n, size=min(10000, n), replace=False) if n > 10000 else np.arange(n)
    Z_sub = fitted.named_steps["preprocess"].transform(X.iloc[idx])
    if hasattr(Z_sub, "toarray"):
        Z_sub = Z_sub.toarray()
    labels_sub = labels[idx]
    sil = silhouette_score(Z_sub, labels_sub, sample_size=min(10000, len(Z_sub)),
                           random_state=42) if len(set(labels_sub)) > 1 else -1.0
    db = davies_bouldin_score(Z_sub, labels_sub)
    ch = calinski_harabasz_score(Z_sub, labels_sub)

    aris = []
    for _ in range(3):
        w = rng.multinomial(n, np.ones(n) / n)
        rep = make_pipe(k)
        rep = mlca_runtime.fit_record(rep, X, "train", kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(labels, rep.predict(X)))
    stab = float(np.mean(aris))
    score = float(sil + 0.1 * stab)
    results.append((score, k, fitted, sil, stab, db, ch))

    for name, val in (("silhouette", sil), ("davies_bouldin", db),
                      ("calinski_harabasz", ch), ("stability", stab),
                      ("selection_score", score)):
        mlca_runtime.record_metric(f"{name}_k{k}", float(val), "train")

best_score, best_k, best_pipeline, best_sil, best_stab, _, _ = max(
    results, key=lambda r: (r[0], r[3], r[4])
)
mlca_runtime.record_metric("selected_k", float(best_k), "train")
mlca_runtime.record_metric("selected_silhouette", float(best_sil), "train")
mlca_runtime.record_metric("selected_stability", float(best_stab), "train")
mlca_runtime.save_model(best_pipeline, "alternative_model.pkl")

print("rows", n, "features", len(input_cols))
print("k_range", min(k_values), max(k_values))
print("selected_k", best_k)
print("silhouette", round(best_sil, 4), "stability", round(best_stab, 4))
