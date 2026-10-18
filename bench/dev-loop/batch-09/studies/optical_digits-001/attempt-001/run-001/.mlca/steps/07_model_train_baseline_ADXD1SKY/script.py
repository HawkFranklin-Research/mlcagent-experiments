import json, pickle
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

targets = protocol.get("targets") or []
excluded = protocol.get("excluded_columns") or []
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())

def key_like(s):
    if len(s) != s.nunique(dropna=False):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        return any(t in str(s.name).lower() for t in ("id", "key", "code", "seq", "index"))
    return False

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_ZVECC4HC/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_SAGMFV47/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def raw_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, s in obj.named_steps.items():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

frontend = None
if sel is not None and raw_cols(sel) == feature_cols:
    frontend = sel
elif pre is not None and raw_cols(pre) == feature_cols:
    frontend = Pipeline([("pre", pre), ("sel", sel)]) if sel is not None else pre

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

n_out = len(feature_cols) if frontend is None else int(frontend.transform(X_train.head(1)).shape[1])
n_comp = min(8, n_out)
if n_comp < 1:
    raise ValueError("No features available for PCA")

steps = [("frontend", frontend)] if frontend is not None else []
steps.append(("pca", PCA(n_components=n_comp)))
model = Pipeline(steps)
model = mlca_runtime.fit_record(model, X_train, "train")
mlca_runtime.save_model(model, "model.pkl")

def recon_mse(m, X):
    Z = m.named_steps["frontend"].transform(X) if "frontend" in m.named_steps else X
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    pca = m.named_steps["pca"]
    R = pca.inverse_transform(pca.transform(Z))
    return float(np.mean((Z - R) ** 2))

train_mse = recon_mse(model, X_train)
val_mse = recon_mse(model, X_val)
mlca_runtime.record_metric("reconstruction_error", train_mse, "train")
mlca_runtime.record_metric("reconstruction_error", val_mse, "val")
mlca_runtime.record_metric("embedding_dimension", float(n_comp), "train")

print(f"rows train={len(X_train)} val={len(X_val)} feats={len(feature_cols)}")
print(f"recon_mse train={train_mse:.6g} val={val_mse:.6g} pca_dim={n_comp}")
