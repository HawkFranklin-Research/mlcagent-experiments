import json
import warnings
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
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

warnings.filterwarnings("ignore")
mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
n = len(train)

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    nun = s.nunique(dropna=True)
    is_textish = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_textish and nun == n:
        continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and nun == n:
        name = str(c).lower()
        if s.is_monotonic_increasing or any(t in name for t in ("id", "code", "key")):
            continue
    feature_cols.append(c)

num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) <= max(1, 0.05 * n):
        cat_cols.append(c)

feature_cols = [c for c in feature_cols if c not in date_cols]
X = train[feature_cols].copy()
width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
use_sparse = width > 300


def build(k):
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
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=use_sparse)),
            ]),
            cat_cols,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    if not use_sparse:
        pre.set_output(transform="pandas")
    return Pipeline([
        ("pre", pre),
        ("km", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)),
    ])


rng = np.random.default_rng(42)
k_max = min(10, n - 1)
results = []
for k in range(2, k_max + 1):
    pipe = build(k)
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.predict(X)
    Xt = fitted.named_steps["pre"].transform(X)
    ss = 10000 if n > 10000 else None
    sil = silhouette_score(Xt, labels, sample_size=ss, random_state=42)
    nlab = len(np.unique(labels))
    db = davies_bouldin_score(Xt, labels) if 1 < nlab < n else np.nan
    ch = calinski_harabasz_score(Xt, labels) if 1 < nlab < n else np.nan
    aris = []
    for _ in range(3):
        w = rng.multinomial(n, np.ones(n) / n)
        fb = mlca_runtime.fit_record(clone(pipe), X, "train", km__sample_weight=w)
        aris.append(adjusted_rand_score(labels, fb.predict(X)))
    results.append({"k": k, "pipe": fitted, "sil": sil, "db": db, "ch": ch, "stab": float(np.mean(aris))})

stable = [r for r in results if r["stab"] >= 0.6]
pool = stable if stable else results
best = max(pool, key=lambda r: (r["sil"], r["stab"]))
mlca_runtime.save_model(best["pipe"], "alternative_model.pkl")
mlca_runtime.record_metric("n_clusters", float(best["k"]), "train")
mlca_runtime.record_metric("silhouette", float(best["sil"]), "train")
mlca_runtime.record_metric("stability_ari", float(best["stab"]), "train")
mlca_runtime.record_metric("davies_bouldin", float(best["db"]), "train")
mlca_runtime.record_metric("calinski_harabasz", float(best["ch"]), "train")
print(f"rows={n} features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"selected k={best['k']} silhouette={best['sil']:.4f} stability={best['stab']:.4f}")
