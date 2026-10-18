import mlca_runtime
import json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from scipy.stats import rankdata

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available keys: {sorted(d.keys())}")
    return d[key]

seed = 42
mlca_runtime.set_seed(seed)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = need(protocol, "targets") or []
excluded = need(protocol, "excluded_columns") or []
group = need(protocol, "group_column")
task = need(protocol, "task")
metric_name = task.get("primary_metric") or "silhouette"
train = pd.read_parquet(inp["train"])

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    n = len(s)
    nun = s.nunique(dropna=True)
    name = str(c).lower()
    if nun == n and n > 1:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s) and any(t in name for t in ("id", "code", "no", "num", "key")):
            continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns")
X = train[feature_cols].copy()
n = len(X)
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
width = len(num_cols) + sum(X[c].nunique(dropna=False) for c in cat_cols)
dense = width <= 300

def make_pre():
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=dense))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), cat_cols))
    pre = ColumnTransformer(transformers, sparse_threshold=0.0 if not dense else 0.3)
    if dense:
        pre.set_output(transform="pandas")
    return pre

k_max = min(10, n - 1)
if k_max < 2:
    raise ValueError("Need at least 2 rows")
rng = np.random.default_rng(seed)
results = []
for k in range(2, k_max + 1):
    pipe = Pipeline([("prep", make_pre()), ("cluster", MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(1024, max(256, n // 10)), n_init="auto"))])
    fitted = mlca_runtime.fit_record(pipe, X, partition="train")
    labels = fitted.predict(X)
    Xt = fitted.named_steps["prep"].transform(X)
    if n > 10000:
        idx = rng.choice(n, size=10000, replace=False)
        Xs = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
        ls = labels[idx]
    else:
        Xs, ls = Xt, labels
    if len(np.unique(ls)) > 1:
        sil = float(silhouette_score(Xs, ls, sample_size=min(10000, len(ls)), random_state=seed))
        db = float(davies_bouldin_score(Xs, ls))
        ch = float(calinski_harabasz_score(Xs, ls))
    else:
        sil, db, ch = -1.0, np.inf, -np.inf
    aris = []
    for b in range(3):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        km = MiniBatchKMeans(n_clusters=k, random_state=seed + b, batch_size=min(1024, max(256, n // 10)), n_init="auto")
        mlca_runtime.fit_record(km, Xt, partition="train", sample_weight=w)
        aris.append(adjusted_rand_score(labels, km.predict(Xt)))
    results.append((k, sil, db, ch, float(np.mean(aris)), fitted))

arr = np.array([r[:5] for r in results], dtype=float)
score = rankdata(-arr[:, 1]) + rankdata(arr[:, 2]) + rankdata(-arr[:, 3]) + rankdata(-arr[:, 4])
best_k, best_sil, best_db, best_ch, best_stab, best_pipe = results[int(np.argmin(score))]
mlca_runtime.save_model(best_pipe, "alternative_model.pkl")
mlca_runtime.record_metric(metric_name, best_sil, "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
mlca_runtime.record_metric("stability", best_stab, "train")
print(f"rows={n} features={len(feature_cols)} dense={dense}")
print(f"best_k={best_k} silhouette={best_sil:.4f} stability={best_stab:.4f}")
