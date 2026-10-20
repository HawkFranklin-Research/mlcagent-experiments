import mlca_runtime, json, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = protocol.get("group_column")
task = need(protocol, "task")
primary = task.get("primary_metric", "silhouette")
train = pd.read_parquet(inp["train"])

for k in ["05_preprocess_fit_preprocessing_JSZBRNKQ/manifest.json", "06_features_retain_features_7T23584P/manifest.json"]:
    json.loads(inp[k].read_text())
with open(inp["05_preprocess_fit_preprocessing_JSZBRNKQ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_7T23584P/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

skip = set(targets) | set(excluded) | ({group} if group else set())

def is_key(c, s):
    n = c.lower()
    if n in {"id", "key", "code", "uuid", "index", "row_id", "rowid"}:
        return True
    if n.endswith(("id", "_id", "key", "_key", "code", "_code", "uuid", "_uuid", "index", "_index")):
        if s.dtype == object or pd.api.types.is_integer_dtype(s.dtype):
            return s.nunique(dropna=True) == len(s)
    return False

feat_cols = [c for c in train.columns if c not in skip and not is_key(c, train[c])]

def raw_ok(obj, cols):
    fi = getattr(obj, "feature_names_in_", None)
    return fi is not None and set(fi).issubset(set(cols))

if raw_ok(pre, feat_cols):
    front = pre
elif raw_ok(sel, feat_cols):
    front = sel
elif hasattr(sel, "transform"):
    front = sel
elif hasattr(pre, "transform"):
    front = pre
else:
    raise TypeError("No usable saved transformer")

if getattr(front, "feature_names_in_", None) is not None:
    fc = list(front.feature_names_in_)
    if set(fc) & skip:
        raise ValueError("Saved transformer expects target/excluded/group column")
    feat_cols = fc

X = train[feat_cols].copy()
front = mlca_runtime.fit_record(front, X, "train")
try:
    front.set_output(transform="pandas")
except Exception:
    pass
Xt = front.transform(X)
is_sparse = sp.issparse(Xt) or (isinstance(Xt, pd.DataFrame) and any(isinstance(dt, pd.SparseDtype) for dt in Xt.dtypes))
if not isinstance(Xt, pd.DataFrame):
    Xt = pd.DataFrame.sparse.from_spmatrix(Xt, index=X.index) if sp.issparse(Xt) else pd.DataFrame(np.asarray(Xt), index=X.index)

scaler = StandardScaler(with_mean=not is_sparse)
try:
    scaler.set_output(transform="pandas")
except Exception:
    pass
scaler = mlca_runtime.fit_record(scaler, Xt, "train")
try:
    scaler.set_output(transform="pandas")
except Exception:
    pass
Xs = scaler.transform(Xt)
if not isinstance(Xs, pd.DataFrame):
    Xs = pd.DataFrame.sparse.from_spmatrix(Xs, index=X.index) if sp.issparse(Xs) else pd.DataFrame(np.asarray(Xs), index=X.index)

def dense(a):
    if isinstance(a, pd.DataFrame):
        if any(isinstance(dt, pd.SparseDtype) for dt in a.dtypes):
            return a.sparse.to_dense().to_numpy()
        return a.to_numpy()
    return a.toarray() if sp.issparse(a) else np.asarray(a)

Xm = dense(Xs)
n = len(X)
ks = [1] if n < 3 else list(range(2, min(10, n - 1) + 1))
rng = np.random.RandomState(123)
results = []

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km = mlca_runtime.fit_record(km, Xs, "train")
    lab = km.labels_
    if k == 1:
        sil, db, ch = np.nan, np.nan, np.nan
    else:
        sil = silhouette_score(Xm, lab, sample_size=min(10000, n), random_state=42)
        db = davies_bouldin_score(Xm, lab)
        ch = calinski_harabasz_score(Xm, lab)
    stabs = []
    if k > 1:
        for b in range(5):
            w = rng.multinomial(n, np.ones(n) / n)
            kmb = KMeans(n_clusters=k, n_init=5, random_state=1000 + b)
            kmb = mlca_runtime.fit_record(kmb, Xs, "train", sample_weight=w)
            stabs.append(adjusted_rand_score(lab, kmb.labels_))
    stab = float(np.mean(stabs)) if stabs else 0.0
    score = (0.0 if np.isnan(sil) else sil) + 0.05 * stab
    results.append((score, k, sil, db, ch, stab))
    if not np.isnan(sil):
        mlca_runtime.record_metric(f"silhouette_k{k}", float(sil), "train")
        mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

best = max(results, key=lambda r: r[0])
_, best_k, best_sil, best_db, best_ch, best_stab = best
mlca_runtime.record_metric(primary, float(best_sil) if not np.isnan(best_sil) else 0.0, "train")
mlca_runtime.record_metric("stability", float(best_stab), "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
mlca_runtime.record_metric("davies_bouldin", float(best_db) if not np.isnan(best_db) else 0.0, "train")
mlca_runtime.record_metric("calinski_harabasz", float(best_ch) if not np.isnan(best_ch) else 0.0, "train")

km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
km_final = mlca_runtime.fit_record(km_final, Xs, "train")
final_pipe = Pipeline([("front", front), ("scale", scaler), ("kmeans", km_final)])
mlca_runtime.save_model(final_pipe, "model.pkl")

print(f"train rows={n} features={len(feat_cols)}")
print(f"selected k={best_k} silhouette={best_sil:.4f} stability={best_stab:.4f}")
