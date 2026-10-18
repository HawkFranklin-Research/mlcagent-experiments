import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing required key {k!r}; available keys: {sorted(d.keys())}")
    return d[k]

targets = req(protocol, "targets") or []
excluded = req(protocol, "excluded_columns") or []
group = req(protocol, "group_column")
task = req(protocol, "task")
primary_metric = task["primary_metric"]

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

def key_like(s):
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == n:
        nm = str(s.name).lower()
        return any(t in nm for t in ("id", "key", "code", "seq", "index"))
    return False

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_ZVECC4HC/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_SAGMFV47/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def raw_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def accepts_raw(obj):
    cols = raw_cols(obj)
    return cols is not None and cols == feature_cols

sel_is_wrapped = isinstance(selector, Pipeline) and len(selector.named_steps) > 1

if selector is not None and sel_is_wrapped and accepts_raw(selector):
    frontend = selector
    fit_cols = raw_cols(selector)
elif preprocessor is not None and accepts_raw(preprocessor):
    fit_cols = raw_cols(preprocessor)
    frontend = preprocessor if selector is None else Pipeline([("preprocessor", preprocessor), ("selector", selector)])
elif selector is not None and accepts_raw(selector):
    frontend = selector
    fit_cols = raw_cols(selector)
elif preprocessor is not None:
    fit_cols = raw_cols(preprocessor) or feature_cols
    frontend = preprocessor if selector is None else Pipeline([("preprocessor", preprocessor), ("selector", selector)])
else:
    raise ValueError("No saved frontend accepts the raw feature DataFrame")

missing = [c for c in fit_cols if c not in train.columns]
if missing:
    raise ValueError(f"Saved frontend expects missing columns: {missing}")
bad = [c for c in fit_cols if c in drop]
if bad:
    raise ValueError(f"Saved frontend includes target/excluded columns: {bad}")

X_train = train[fit_cols].copy()
X_val = val[fit_cols].copy()

probe = frontend.transform(X_train.head(1))
n_out = probe.shape[1]
n_components = min(8, n_out)
if n_components < 1:
    raise ValueError("No features available for PCA")

model = Pipeline([("frontend", frontend), ("pca", PCA(n_components=n_components))])
model = mlca_runtime.fit_record(model, X_train, "train")
mlca_runtime.save_model(model, "model.pkl")

def recon_mse(m, X):
    fe = m.named_steps["frontend"]
    pca = m.named_steps["pca"]
    Z = fe.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    Z = np.asarray(Z)
    Zp = pca.transform(Z)
    Zr = pca.inverse_transform(Zp)
    return float(np.mean((Z - Zr) ** 2))

train_mse = recon_mse(model, X_train)
val_mse = recon_mse(model, X_val)
mlca_runtime.record_metric(primary_metric, train_mse, "train")
mlca_runtime.record_metric(primary_metric, val_mse, "val")
mlca_runtime.record_metric("embedding_dimension", float(n_components), "train")

print(f"train rows={len(X_train)} features={len(fit_cols)} pca_components={n_components}")
print(f"{primary_metric} (frontend space): train={train_mse:.6g} val={val_mse:.6g}")
