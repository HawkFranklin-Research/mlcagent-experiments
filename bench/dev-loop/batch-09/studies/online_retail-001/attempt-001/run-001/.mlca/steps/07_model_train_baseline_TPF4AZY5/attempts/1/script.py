import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"MLCA: protocol missing {key}; available {list(protocol)}")
targets = list(protocol.get("targets") or [])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
task = protocol["task"]
seed = 42

train = pd.read_parquet(inp["train"])
n = len(train)
if n == 0:
    raise ValueError("MLCA: empty training table")

def keylike(s):
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    if not (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return False
    return s.notna().all() and s.nunique(dropna=True) == len(s)

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if keylike(train[c]):
        continue
    feat_cols.append(c)

with open(inp["05_preprocess_fit_preprocessing_MW2B6JEQ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_JF8SWM3Y/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def names(obj):
    try:
        return list(obj.feature_names_in_)
    except Exception:
        return None

pre_in = names(pre)
sel_in = names(sel)
pre_out = None
if pre is not None:
    try:
        pre_out = list(pre.get_feature_names_out())
    except Exception:
        pre_out = None
    if pre_out is not None and len(pre_out) <= 300:
        try:
            pre.set_output(transform="pandas")
        except Exception:
            pass

sel_steps = list(sel.named_steps) if isinstance(sel, Pipeline) else []
pre_included = any(s in sel_steps for s in ("preprocessor", "preprocess", "prep", "preprocessing"))
if sel is not None and pre_included:
    frontend = sel
elif sel is not None and sel_in is not None and all(c in feat_cols for c in sel_in):
    frontend = sel
elif pre is not None and sel is not None and pre_out is not None and sel_in is not None and list(sel_in) == list(pre_out):
    frontend = Pipeline([("pre", pre), ("selector", sel)])
elif pre is not None and sel is not None:
    frontend = sel if isinstance(sel, Pipeline) and len(sel.named_steps) > 1 else Pipeline([("pre", pre), ("selector", sel)])
elif pre is not None:
    frontend = pre
elif sel is not None:
    frontend = sel
else:
    raise ValueError("MLCA: no preprocessing or selector input available")

in_cols = names(frontend)
if not in_cols and isinstance(frontend, Pipeline) and frontend.steps:
    in_cols = names(frontend.steps[0][1])
if in_cols and all(c in train.columns for c in in_cols):
    if any(c in targets or c in excluded for c in in_cols):
        raise ValueError("MLCA: target/excluded column in frontend input")
    X = train.loc[:, in_cols].copy()
else:
    X = train.loc[:, feat_cols].copy()

def make_pipe(k):
    return Pipeline([
        ("frontend", clone(frontend)),
        ("kmeans", KMeans(n_clusters=k, random_state=seed, n_init=10)),
    ])

ks = range(2, 9)
base = {}
rows = []
rng = np.random.default_rng(seed)
for k in ks:
    fitted = mlca_runtime.fit_record(make_pipe(k), X, "train")
    labels = fitted.predict(X)
    Xt = fitted.named_steps["frontend"].transform(X)
    if hasattr(Xt, "toarray") and getattr(Xt, "shape", (0, 0))[1] <= 300:
        Xt_metric = Xt.toarray()
    else:
        Xt_metric = Xt
    sil = float(silhouette_score(Xt_metric, labels, sample_size=min(10000, n), random_state=seed))
    try:
        db = float(davies_bouldin_score(Xt_metric, labels))
    except Exception:
        db = float("nan")
    try:
        ch = float(calinski_harabasz_score(Xt_metric, labels))
    except Exception:
        ch = float("nan")
    aris = []
    for _ in range(5):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        f2 = mlca_runtime.fit_record(make_pipe(k), X, "train", kmeans__sample_weight=w)
        aris.append(float(adjusted_rand_score(labels, f2.predict(X))))
    stab = float(np.mean(aris)) if aris else 0.0
    base[k] = fitted
    rows.append((k, sil, db, ch, stab))
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")
    mlca_runtime.record_metric(f"davies_bouldin_k{k}", db, "train")
    mlca_runtime.record_metric(f"calinski_harabasz_k{k}", ch, "train")

valid = [r for r in rows if np.isfinite(r[1])]
if not valid:
    valid = rows
best = max(valid, key=lambda r: r[1] + 0.25 * r[4])
selected_k = best[0]
model = base[selected_k]
mlca_runtime.save_model(model, "model.pkl")
mlca_runtime.record_metric("selected_clusters", float(selected_k), "train")
mlca_runtime.record_metric("selected_silhouette", float(best[1]), "train")
mlca_runtime.record_metric("selected_stability", float(best[4]), "train")
print(f"train rows={n} features={X.shape[1]}")
print(f"candidates={list(ks)}")
print(f"selected k={selected_k} silhouette={best[1]:.4f} stability={best[4]:.4f}")
