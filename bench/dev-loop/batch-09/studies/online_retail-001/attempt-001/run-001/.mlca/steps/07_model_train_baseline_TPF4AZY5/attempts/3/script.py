import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
P = json.loads(inp["_protocol"].read_text())
targets = list(P.get("targets") or [])
excluded = list(P.get("excluded_columns") or [])
group = P.get("group_column")

train = pd.read_parquet(inp["train"])
n = len(train)

def keylike(s):
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    if not (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return False
    return s.notna().all() and s.nunique(dropna=True) == len(s)

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and not keylike(train[c])]

with open(inp["05_preprocess_fit_preprocessing_MW2B6JEQ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_JF8SWM3Y/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def names(o):
    try:
        return list(o.feature_names_in_)
    except Exception:
        return None

pre_in, sel_in = names(pre), names(sel)
pre_out = None
if pre is not None:
    try:
        pre_out = list(pre.get_feature_names_out())
        if len(pre_out) <= 300:
            pre.set_output(transform="pandas")
    except Exception:
        pre_out = None

sel_steps = list(sel.named_steps) if isinstance(sel, Pipeline) else []
pre_in_sel = any(s in sel_steps for s in ("preprocessor", "preprocess", "prep", "preprocessing"))

if sel is not None and pre_in_sel:
    frontend = sel
elif sel is not None and sel_in is not None and all(c in feat_cols for c in sel_in):
    frontend = sel
elif pre is not None and sel is not None and pre_out is not None and sel_in is not None and list(sel_in) == list(pre_out):
    frontend = Pipeline([("pre", pre), ("selector", sel)])
elif pre is not None and sel is not None:
    frontend = Pipeline([("pre", pre), ("selector", sel)])
elif pre is not None:
    frontend = pre
elif sel is not None:
    frontend = sel
else:
    raise ValueError("MLCA: no preprocessing or selector input")

in_cols = names(frontend)
X = train.loc[:, in_cols].copy() if in_cols and all(c in train.columns for c in in_cols) else train.loc[:, feat_cols].copy()
for c in X.columns:
    if c in targets or c in excluded:
        raise ValueError("MLCA: target/excluded in feature frame")

def make_pipe(k):
    return Pipeline([("frontend", clone(frontend)), ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))])

ks = list(range(2, 9))
rng = np.random.default_rng(42)
base, rows = {}, []
for k in ks:
    fitted = mlca_runtime.fit_record(make_pipe(k), X, "train")
    labels = fitted.predict(X)
    Xt = fitted.named_steps["frontend"].transform(X)
    Xt_m = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    sil = float(silhouette_score(Xt_m, labels, sample_size=min(10000, n), random_state=42))
    db = float(davies_bouldin_score(Xt_m, labels))
    ch = float(calinski_harabasz_score(Xt_m, labels))
    aris = []
    for _ in range(3):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        f2 = mlca_runtime.fit_record(make_pipe(k), X, "train", kmeans__sample_weight=w)
        aris.append(float(adjusted_rand_score(labels, f2.predict(X))))
    stab = float(np.mean(aris))
    base[k] = fitted
    rows.append((k, sil, db, ch, stab))
    for nm, v in (("silhouette", sil), ("stability", stab), ("davies_bouldin", db), ("calinski_harabasz", ch)):
        mlca_runtime.record_metric(f"{nm}_k{k}", v, "train")

valid = [r for r in rows if np.isfinite(r[1])] or rows
best = max(valid, key=lambda r: r[1] + 0.25 * r[4])
mlca_runtime.save_model(base[best[0]], "model.pkl")
mlca_runtime.record_metric("selected_clusters", float(best[0]), "train")
mlca_runtime.record_metric("selected_silhouette", float(best[1]), "train")
mlca_runtime.record_metric("selected_stability", float(best[4]), "train")
print(f"train rows={n} features={X.shape[1]}")
print(f"selected k={best[0]} sil={best[1]:.4f} stab={best[4]:.4f}")
