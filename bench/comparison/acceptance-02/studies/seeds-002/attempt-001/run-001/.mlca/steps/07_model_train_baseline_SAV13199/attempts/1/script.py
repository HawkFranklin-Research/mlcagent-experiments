import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol.get("targets") or []
excluded = protocol.get("excluded_columns") or []
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
n = len(train)
drop_cols = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop_cols]

keylike = set()
for c in feat_cols:
    s = train[c].dropna()
    if len(s) == 0 or s.nunique() < len(s):
        continue
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        keylike.add(c)
feat_cols = [c for c in feat_cols if c not in keylike]

num_cols, low_cat = [], []
for c in feat_cols:
    col = train[c]
    if pd.api.types.is_datetime64_any_dtype(col):
        continue
    if pd.api.types.is_numeric_dtype(col):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(col) or isinstance(col.dtype, pd.CategoricalDtype):
        if col.nunique() <= max(20, int(0.05 * n)):
            low_cat.append(c)

pre = None
for key in ["06_features_retain_features_4JEKVG72/selector_all.pkl",
            "05_preprocess_fit_preprocessing_SYRRSV3B/preprocessor.pkl"]:
    try:
        with open(paths[key], "rb") as f:
            cand = pickle.load(f)
        fni = getattr(cand, "feature_names_in_", None)
        if fni is not None and set(map(str, fni)) == set(feat_cols):
            pre = cand
            feat_cols = list(map(str, fni))
            break
    except Exception:
        continue

if pre is None:
    tfs = []
    if num_cols:
        tfs.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("sc", StandardScaler())]), num_cols))
    if low_cat:
        tfs.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                    low_cat))
    pre = ColumnTransformer(tfs, remainder="drop")
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

results = []
k_max = min(10, max(2, n // 20))
for k in range(2, k_max + 1):
    pipe = Pipeline([("pre", pre), ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])
    fitted = mlca_runtime.fit_record(pipe, train[feat_cols], "train")
    Xt = fitted.named_steps["pre"].transform(train[feat_cols])
    Xa = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    labels = fitted.named_steps["km"].labels_
    try:
        sil = float(silhouette_score(Xa, labels, sample_size=min(10000, len(Xa)), random_state=42))
        db = float(davies_bouldin_score(Xa, labels))
        ch = float(calinski_harabasz_score(Xa, labels))
    except Exception:
        continue
    rng_b = np.random.RandomState(42)
    aris = []
    for b in range(5):
        idx = rng_b.randint(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        pipe_b = Pipeline([("pre", pre), ("km", KMeans(n_clusters=k, n_init=5, random_state=42 + b))])
        fb = mlca_runtime.fit_record(pipe_b, train[feat_cols], "train", **{"km__sample_weight": w})
        lb = fb.named_steps["km"].labels_
        aris.append(float(adjusted_rand_score(labels, lb)))
    results.append(dict(k=k, sil=sil, db=db, ch=ch, stab=float(np.mean(aris))))

scores = [r["sil"] * max(r["stab"], 0.05) for r in results]
best = results[int(np.argmax(scores))]
final = Pipeline([("pre", pre), ("km", KMeans(n_clusters=best["k"], n_init=20, random_state=42))])
final_fit = mlca_runtime.fit_record(final, train[feat_cols], "train")
mlca_runtime.save_model(final_fit, "model.pkl")

mlca_runtime.record_metric("selected_k", float(best["k"]), "train")
mlca_runtime.record_metric("silhouette", best["sil"], "train")
mlca_runtime.record_metric("davies_bouldin", best["db"], "train")
mlca_runtime.record_metric("calinski_harabasz", best["ch"], "train")
mlca_runtime.record_metric("bootstrap_stability", best["stab"], "train")

tr = pd.DataFrame({"row_id": train.index, "partition": "train",
                   "cluster": final_fit.predict(train[feat_cols]).astype(int)})
mlca_runtime.save_row_outputs(tr, "train_clusters.parquet")
vr = pd.DataFrame({"row_id": val.index, "partition": "val",
                   "cluster": final_fit.predict(val[feat_cols]).astype(int)})
mlca_runtime.save_row_outputs(vr, "val_clusters.parquet")

print("k=", best["k"], "sil=", round(best["sil"], 4), "stab=", round(best["stab"], 4))
