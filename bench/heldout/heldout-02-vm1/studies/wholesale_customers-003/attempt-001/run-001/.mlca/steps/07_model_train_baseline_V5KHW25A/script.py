import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
try:
    targets = list(proto["targets"])
    excluded = list(proto["excluded_columns"])
    task = proto["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto.keys())}")
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
n = len(train)

def key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=False) == n
    name = str(s.name).lower()
    return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == n and any(
        t in name for t in ("id", "key", "index", "row")
    )

base = [
    c for c in train.columns
    if c not in set(targets) | set(excluded)
    and c != group
    and not key_like(train[c])
]
if not base:
    raise ValueError("No usable feature columns")

def input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline) and obj.steps:
        return input_names(obj.steps[0][1])
    return None

def names_ok(obj):
    nms = input_names(obj)
    return nms is not None and set(nms).issubset(set(base))

def has_scaler(obj):
    if isinstance(obj, Pipeline):
        return any(has_scaler(v) for _, v in obj.steps)
    if isinstance(obj, ColumnTransformer):
        return any(has_scaler(t) for _, t, _ in obj.transformers)
    return "Scaler" in obj.__class__.__name__

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_HBN850GP/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_5BEJHTH5/selector_all.pkl"])

front = None
if sel is not None and isinstance(sel, Pipeline) and names_ok(sel):
    front = sel
elif pre is not None and names_ok(pre):
    front = pre
elif sel is not None and names_ok(sel):
    front = sel

steps = []
if front is not None:
    steps.append(("features", front))
    if front is pre and sel is not None:
        if isinstance(sel, Pipeline):
            if not names_ok(sel):
                steps.append(("selector", sel))
        elif (not names_ok(sel)) or input_names(sel) == input_names(pre):
            steps.append(("selector", sel))
    if not any(has_scaler(s) for _, s in steps):
        steps.append(("scale", StandardScaler(with_mean=False)))
    input_cols = input_names(front) or base
else:
    num = [c for c in base if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
    cat = [c for c in base if c not in num and not pd.api.types.is_datetime64_any_dtype(train[c])]
    cat = [c for c in cat if train[c].nunique(dropna=False) <= max(20, int(0.05 * n))]
    approx = len(num) + sum(train[c].nunique(dropna=False) for c in cat)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=(approx > 300))
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", oh)]), cat),
    ], remainder="drop")
    if approx <= 300:
        ct.set_output(transform="pandas")
    steps.append(("features", ct))
    input_cols = base

base_pipe = Pipeline(steps + [("kmeans", KMeans(n_clusters=2, random_state=42, n_init=10))])
X = train[list(input_cols)]
max_k = min(8, n - 1)
rng = np.random.default_rng(42)
counts = rng.multinomial(n, np.ones(n) / n)

results = []
for k in range(2, max_k + 1):
    pipe = clone(base_pipe)
    pipe.set_params(kmeans__n_clusters=k)
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.predict(X)

    Xt = X
    for _, st in fitted.steps[:-1]:
        Xt = st.transform(Xt)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    sil = float(silhouette_score(Xt, labels, sample_size=min(10000, n - 1), random_state=42))
    db = float(davies_bouldin_score(Xt, labels))
    ch = float(calinski_harabasz_score(Xt, labels))

    aris = []
    for _ in range(5):
        bp = clone(fitted)
        bp = mlca_runtime.fit_record(bp, X, "train", kmeans__sample_weight=counts)
        aris.append(adjusted_rand_score(labels, bp.predict(X)))
    stab = float(np.mean(aris))

    score = sil + 0.05 * stab
    results.append((score, k, sil, stab, db, ch, fitted))
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")
    mlca_runtime.record_metric(f"davies_bouldin_k{k}", db, "train")
    mlca_runtime.record_metric(f"calinski_harabasz_k{k}", ch, "train")

best = max(results, key=lambda r: r[0])
mlca_runtime.save_model(best[6], "model.pkl")
mlca_runtime.record_metric("selected_k", float(best[1]), "train")
mlca_runtime.record_metric("selected_silhouette", best[2], "train")

print(f"rows={n} features={len(base)} k_range=2-{max_k}")
print(f"selected_k={best[1]} silhouette={best[2]:.4f} stability={best[3]:.4f}")
print("saved model.pkl")
