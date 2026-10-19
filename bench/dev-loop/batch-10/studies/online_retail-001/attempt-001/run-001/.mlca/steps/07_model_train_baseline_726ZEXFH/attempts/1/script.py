import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
mlca_runtime.set_seed(SEED)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]
df = pd.read_parquet(paths["train"])
n = len(df)
exclude = set(targets) | set(excluded) | ({group} if group else set())

feat_cols = []
for c in df.columns:
    if c in exclude:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    nu = s.nunique(dropna=False)
    if pd.api.types.is_integer_dtype(s) and nu == n and n > 2:
        continue
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and (nu == n or nu > 50):
        continue
    feat_cols.append(c)

if not feat_cols:
    raise ValueError("No usable feature columns after protocol exclusions")

num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]
width = len(num_cols) + sum(int(df[c].nunique(dropna=False)) for c in cat_cols)
use_pd = width <= 300

trans = []
if num_cols:
    trans.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
if cat_cols:
    trans.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pd))]), cat_cols))

X = df[feat_cols].copy()
rng = np.random.default_rng(SEED)
max_k = min(8, n - 1)
ks = list(range(2, max_k + 1))
if not ks:
    raise ValueError("Need at least 3 rows for clustering")

scores = {}
for k in ks:
    pre = ColumnTransformer(trans, remainder="drop", verbose_feature_names_out=False)
    pipe = Pipeline([("pre", pre), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=SEED))])
    if use_pd:
        pipe.set_output(transform="pandas")
    pipe = mlca_runtime.fit_record(pipe, X, "train")
    Xt = pipe.named_steps["pre"].transform(X)
    labels = pipe.named_steps["kmeans"].labels_
    idx = rng.choice(n, size=min(n, 10000), replace=False)
    Xm = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
    if hasattr(Xm, "toarray"):
        Xm = Xm.toarray()
    lm = labels[idx]
    sil = float(silhouette_score(Xm, lm, sample_size=len(Xm), random_state=SEED))
    db = float(davies_bouldin_score(Xm, lm))
    ch = float(calinski_harabasz_score(Xm, lm))
    aris = []
    for b in range(5):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        km = mlca_runtime.fit_record(
            KMeans(n_clusters=k, n_init=5, random_state=SEED + b),
            Xt,
            "train",
            sample_weight=w,
        )
        aris.append(adjusted_rand_score(labels, km.labels_))
    stab = float(np.mean(aris))
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"davies_bouldin_k{k}", db, "train")
    mlca_runtime.record_metric(f"calinski_harabasz_k{k}", ch, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")
    scores[k] = (sil, stab, pipe)

best_k = max(scores, key=lambda k: (round(scores[k][0], 6), round(scores[k][1], 6)))
best_pipe = scores[best_k][2]
mlca_runtime.record_metric("selected_k", float(best_k), "train")
mlca_runtime.record_metric("silhouette_selected", scores[best_k][0], "train")
mlca_runtime.save_model(best_pipe, "model.pkl")
print(f"rows={n}, features={len(feat_cols)}, numeric={len(num_cols)}, categorical={len(cat_cols)}")
print(f"metric_sample={min(n, 10000)}, k_range={ks[0]}-{ks[-1]}, selected_k={best_k}")
print(f"selected_silhouette={scores[best_k][0]:.4f}, stability={scores[best_k][1]:.4f}")
