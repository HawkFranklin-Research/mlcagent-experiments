import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing {e}; available top-level keys: {list(protocol)}") from e
group = protocol.get("group_column", None)
metric = task.get("primary_metric", "silhouette")
train = pd.read_parquet(inp["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    name = str(c).lower()
    if s.nunique(dropna=False) == len(s) and (
        s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category"
        or name in {"id", "row_id", "index", "key"} or name.endswith("_id")
    ):
        continue
    feature_cols.append(c)

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_1E8V578C/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_retain_features_Y7PF8JJ2/selector_all.pkl"].read_bytes())

def first_step(o):
    while isinstance(o, Pipeline):
        o = o.steps[0][1]
    return o

def raw_names(o):
    return getattr(first_step(o), "feature_names_in_", None) if o is not None else None

def is_raw(o):
    names = raw_names(o)
    return names is not None and set(map(str, names)).issubset(set(map(str, feature_cols)))

frontend = None
if sel is not None and is_raw(sel):
    frontend = sel
elif pre is not None and is_raw(pre):
    frontend = Pipeline([("preprocessor", clone(pre)), ("selector", clone(sel))]) if sel is not None else pre
elif pre is not None:
    frontend = pre
elif sel is not None:
    frontend = sel
else:
    raise ValueError("No saved preprocessing or selector available")

expected = raw_names(frontend)
if expected is not None:
    expected = list(expected)
    bad = set(map(str, expected)) & set(map(str, drop))
    if bad:
        raise ValueError(f"Saved frontend expects excluded columns: {sorted(bad)}")
    X = train.loc[:, expected].copy()
else:
    X = train.loc[:, feature_cols].copy()

n = len(X)
if n < 3:
    raise ValueError("Need at least 3 rows for clustering")
max_k = min(10, n - 1)
batch = max(2, min(1024, n))

def make_pipe(k):
    return Pipeline([
        ("frontend", clone(frontend)),
        ("kmeans", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=batch)),
    ])

cand = []
for k in range(2, max_k + 1):
    fit = mlca_runtime.fit_record(make_pipe(k), X, "train")
    labels = fit.predict(X)
    if len(np.unique(labels)) < 2:
        continue
    Xt = fit.named_steps["frontend"].transform(X)
    ss = min(10000, n)
    sil = silhouette_score(Xt, labels, sample_size=ss, random_state=42)
    db = davies_bouldin_score(Xt, labels)
    rng = np.random.default_rng(42 + k)
    aris = []
    for _ in range(5):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        fb = mlca_runtime.fit_record(make_pipe(k), X, "train", kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(labels, fb.predict(X)))
    cand.append((k, sil, db, float(np.mean(aris))))

if not cand:
    raise RuntimeError("No valid cluster candidate")
best_k, best_sil, best_db, best_stab = max(cand, key=lambda r: (r[1], r[3]))
final = mlca_runtime.fit_record(make_pipe(best_k), X, "train")
mlca_runtime.save_model(final, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", float(best_sil), "train")
mlca_runtime.record_metric("davies_bouldin", float(best_db), "train")
mlca_runtime.record_metric("stability_ari", float(best_stab), "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
print(f"train rows={n} features={X.shape[1]}")
print(f"best k={best_k} silhouette={best_sil:.4f} stability={best_stab:.4f}")
