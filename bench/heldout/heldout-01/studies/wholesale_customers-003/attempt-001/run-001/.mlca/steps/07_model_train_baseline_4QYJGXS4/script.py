import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
train = pd.read_parquet(inp["train"])
required = set(targets) | set(excluded) | ({group} if group else set())
raw_cols = [c for c in train.columns if c not in required]

def keylike(c):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s):
        return (s.nunique(dropna=True) == len(s) and len(s) > 2
                and any(t in c.lower() for t in ("id", "key", "code", "index")))
    return False

raw_cols = [c for c in raw_cols if not keylike(c)]
X = train[raw_cols].copy()

with open(inp["05_preprocess_fit_preprocessing_TEP20BJ6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_HRWJ3W8S/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(o):
    n = getattr(o, "feature_names_in_", None)
    if n is not None:
        return list(n)
    if isinstance(o, Pipeline) and o.steps:
        n = getattr(o.steps[0][1], "feature_names_in_", None)
        if n is not None:
            return list(n)
    return None

def accepts_raw(o):
    n = input_names(o)
    return n is not None and list(n) == raw_cols

def out_names(o):
    try:
        return list(o.get_feature_names_out())
    except Exception:
        return None

steps = []
if accepts_raw(sel) and isinstance(sel, Pipeline):
    steps.append(("features", sel))
elif accepts_raw(pre):
    steps.append(("preprocess", pre))
    if not accepts_raw(sel):
        po, si = out_names(pre), input_names(sel)
        if po is not None and si is not None and set(si).issubset(set(po)):
            steps.append(("select", sel))
elif accepts_raw(sel):
    steps.append(("features", sel))
else:
    num = [c for c in raw_cols if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in raw_cols if c not in num and
           train[c].nunique(dropna=True) <= max(5, int(0.05 * len(train)))]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    ct = ColumnTransformer(transformers, remainder="drop")
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass
    steps.append(("preprocess", ct))

steps.append(("scaler", StandardScaler(with_mean=False)))
n = len(X)
if n < 3:
    raise ValueError("Need at least 3 training rows")
max_k = min(8, n - 1)
results = []
for k in range(2, max_k + 1):
    est = (KMeans(n_clusters=k, random_state=42, n_init=10) if n <= 100000
           else MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=1024))
    pipe = Pipeline(steps + [("kmeans", est)])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.predict(X)
    Z = Pipeline(steps).transform(X)
    idx = np.arange(n)
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=10000, replace=False))
    Z_eval = Z.iloc[idx] if hasattr(Z, "iloc") else Z[idx]
    if hasattr(Z_eval, "toarray"):
        Z_eval = Z_eval.toarray()
    sil = (silhouette_score(Z_eval, labels[idx], sample_size=len(idx), random_state=42)
           if len(set(labels[idx])) > 1 and len(idx) > k else -1.0)
    db = (davies_bouldin_score(Z_eval, labels[idx])
          if len(set(labels[idx])) > 1 else np.inf)
    ch = (calinski_harabasz_score(Z_eval, labels[idx])
          if len(set(labels[idx])) > 1 else -1.0)
    aris = []
    for b in range(5):
        rng_i = np.random.default_rng(1000 + b)
        w = rng_i.multinomial(n, np.ones(n) / n).astype(float)
        est2 = (KMeans(n_clusters=k, random_state=42 + b, n_init=10) if n <= 100000
                else MiniBatchKMeans(n_clusters=k, random_state=42 + b, n_init=10, batch_size=1024))
        pipe2 = Pipeline(steps + [("kmeans", est2)])
        boot = mlca_runtime.fit_record(pipe2, X, "train", kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(labels, boot.predict(X)))
    stab = float(np.mean(aris))
    results.append((k, sil, db, ch, stab, fitted))
    mlca_runtime.record_metric(f"silhouette_k{k}", float(sil), "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

stable = [r for r in results if r[4] >= 0.75]
pool = stable if stable else results
k, sil, db, ch, stab, model = max(pool, key=lambda r: (r[1], r[4]))
mlca_runtime.record_metric("selected_k", float(k), "train")
mlca_runtime.record_metric("selected_silhouette", float(sil), "train")
mlca_runtime.record_metric("selected_stability", float(stab), "train")
mlca_runtime.record_metric("selected_davies_bouldin", float(db), "train")
mlca_runtime.record_metric("selected_calinski_harabasz", float(ch), "train")
mlca_runtime.save_model(model, "model.pkl")
print(f"rows={n} features={len(raw_cols)}")
print(f"selected_k={k} silhouette={sil:.4g} stability={stab:.4g}")
