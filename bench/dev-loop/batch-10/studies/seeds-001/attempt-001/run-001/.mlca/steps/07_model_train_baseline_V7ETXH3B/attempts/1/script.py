import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column"):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available: {sorted(protocol.keys())}")
targets = list(protocol["targets"] or [])
excluded = list(protocol["excluded_columns"] or [])
group = protocol["group_column"]
forbidden = set(targets) | set(excluded) | ({group} if group else set())

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_6DN7THMT/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_AKTAEF6F/selector_all.pkl"])

def names(obj):
    v = getattr(obj, "feature_names_in_", None)
    return list(v) if v is not None else None

pre_names, sel_names = names(pre), names(sel)

def raw_names(nms):
    return nms is not None and all(c in train.columns for c in nms)

if raw_names(pre_names):
    feature_cols = pre_names
elif raw_names(sel_names):
    feature_cols = sel_names
else:
    feature_cols = [c for c in train.columns if c not in forbidden]
    for c in list(feature_cols):
        s = train[c]
        if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)

bad = sorted(set(feature_cols) & forbidden)
if bad:
    raise ValueError(f"Feature contract contains forbidden columns: {bad}")
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Feature columns missing from train: {missing}")

pre_ok = raw_names(pre_names) and set(pre_names) == set(feature_cols)
sel_ok = raw_names(sel_names) and set(sel_names) == set(feature_cols)

if isinstance(sel, Pipeline) and len(sel.named_steps) >= 2 and sel_ok:
    frontend = sel
elif pre_ok:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)]) if sel is not None else pre
elif sel_ok:
    frontend = sel
else:
    frontend = pre if pre is not None else sel
if frontend is None:
    raise ValueError("No usable saved frontend found")

def width_of(obj):
    if hasattr(obj, "get_support"):
        try:
            return int(np.sum(obj.get_support()))
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for step in reversed(list(obj.named_steps.values())):
            if hasattr(step, "get_support"):
                try:
                    return int(np.sum(step.get_support()))
                except Exception:
                    pass
        try:
            return len(obj.get_feature_names_out())
        except Exception:
            pass
    try:
        return len(obj.get_feature_names_out())
    except Exception:
        pass
    return None

n_feat = width_of(frontend) or len(feature_cols)
n_components = min(8, n_feat)
pipe = Pipeline([
    ("frontend", frontend),
    ("pca", PCA(n_components=n_components, random_state=42)),
])
fitted = mlca_runtime.fit_record(pipe, train[feature_cols], partition="train")
mlca_runtime.save_model(fitted, "model.pkl")

Xv = val[feature_cols]
Z = fitted.named_steps["frontend"].transform(Xv)
if issparse(Z):
    Z = Z.toarray()
pca = fitted.named_steps["pca"]
R = pca.inverse_transform(pca.transform(Z))
err = float(np.mean((np.asarray(Z) - np.asarray(R)) ** 2))
metric = protocol.get("task", {}).get("primary_metric", "reconstruction_error")
mlca_runtime.record_metric(metric, err, split="val")
print(f"features={len(feature_cols)} pca_components={n_components} val_{metric}={err:.4g}")
