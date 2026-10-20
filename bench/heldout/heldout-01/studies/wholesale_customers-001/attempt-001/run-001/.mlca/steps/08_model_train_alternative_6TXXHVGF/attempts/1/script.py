import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
SEED = 42
inputs = mlca_runtime.inputs()
prot = json.loads(inputs["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
miss = [k for k in need if k not in prot]
if miss:
    raise KeyError(f"Missing protocol keys {miss}; available top-level keys: {sorted(prot)}")
targets, excluded, group = prot["targets"], prot["excluded_columns"], prot["group_column"]
X = pd.read_parquet(inputs["train"])

def keylike(name, s):
    n = len(s)
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == n:
        if any(t in str(name).lower() for t in ("id", "key", "code")):
            return True
        v = s.dropna().sort_values().to_numpy()
        return len(v) == n and v.min() == 0 and v.max() == n - 1
    return False

protected = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in X.columns if c not in protected and not keylike(c, X[c])]
X_train = X[feature_cols].copy()
num_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = X_train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_cols.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        (drop_cols if s.nunique(dropna=True) > 0.05 * len(X_train) else cat_cols).append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        u = set(s.dropna().unique())
        (cat_cols if len(u) <= 10 and not u.issubset({0, 1}) else num_cols).append(c)
    else:
        num_cols.append(c)

def make_pipe(k):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]).set_output(transform="pandas")
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]).set_output(transform="pandas")
    trans = []
    if num_cols:
        trans.append(("num", num_pipe, num_cols))
    if cat_cols:
        trans.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(trans, remainder="drop").set_output(transform="pandas")
    return Pipeline([
        ("pre", pre),
        ("km", MiniBatchKMeans(
            n_clusters=k, random_state=SEED, n_init=10,
            batch_size=min(1024, len(X_train))
        )),
    ]).set_output(transform="pandas")

max_k = min(8, max(2, len(X_train) - 1))
ks = list(range(2, max_k + 1))
rows = []
for k in ks:
    pipe = make_pipe(k)
    fitted = mlca_runtime.fit_record(pipe, X_train, "train")
    Z = fitted.named_steps["pre"].transform(X_train)
    lab = fitted.named_steps["km"].labels_
    if len(set(lab)) < 2:
        continue
    sil = silhouette_score(Z, lab, sample_size=min(10000, len(X_train)), random_state=SEED)
    db = davies_bouldin_score(Z, lab)
    ch = calinski_harabasz_score(Z, lab)
    aris = []
    rng = np.random.default_rng(SEED + k)
    for _ in range(5):
        w = rng.multinomial(len(X_train), np.ones(len(X_train)) / len(X_train)).astype(float)
        fb = mlca_runtime.fit_record(clone(pipe), X_train, "train", km__sample_weight=w)
        aris.append(adjusted_rand_score(lab, fb.named_steps["km"].labels_))
    stab = float(np.mean(aris))
    rows.append((k, sil, db, ch, stab, sil + stab, fitted))
    mlca_runtime.record_metric(f"silhouette_k{k}", float(sil), "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

if not rows:
    raise RuntimeError("No valid clustering candidate")
k, sil, db, ch, stab, score, model = max(rows, key=lambda r: r[5])
mlca_runtime.record_metric("selected_k", float(k), "train")
mlca_runtime.record_metric("silhouette", float(sil), "train")
mlca_runtime.record_metric("stability", float(stab), "train")
mlca_runtime.save_model(model, "alternative_model.pkl")
print(f"rows={len(X_train)} features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)} dropped={len(drop_cols)}")
print(f"selected_k={k} silhouette={sil:.4f} stability={stab:.4f}")
