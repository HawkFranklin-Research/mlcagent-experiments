import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_RDQ93D4J/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_0YD36CMR/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def contains(pipe, obj):
    if not isinstance(pipe, Pipeline):
        return False
    return any(step is obj for step in pipe.named_steps.values())

if isinstance(sel, Pipeline) and contains(sel, pre):
    frontend = sel
elif isinstance(pre, Pipeline) and contains(pre, sel):
    frontend = pre
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

if hasattr(frontend, "feature_names_in_"):
    saved_cols = list(frontend.feature_names_in_)
    if set(saved_cols) != set(feat_cols):
        raise ValueError("Feature columns mismatch with saved frontend")
    feat_cols = saved_cols

X = train[feat_cols]
Xv = val[feat_cols]

Xt = frontend.transform(X.head(5))
n_feat = Xt.shape[1] if hasattr(Xt, "shape") else np.asarray(Xt).shape[1]
n_comp = min(8, n_feat)

full = Pipeline([("frontend", frontend), ("pca", PCA(n_components=n_comp, random_state=42))])
mlca_runtime.fit_record(full, X, partition="train")

def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)

pca = full.named_steps["pca"]
fe = full.named_steps["frontend"]

Xt_train = to_dense(fe.transform(X))
Xr_train = pca.inverse_transform(pca.transform(Xt_train))
err_train = float(np.mean((Xt_train - Xr_train) ** 2))

Xt_val = to_dense(fe.transform(Xv))
Xr_val = pca.inverse_transform(pca.transform(Xt_val))
err_val = float(np.mean((Xt_val - Xr_val) ** 2))

mlca_runtime.record_metric("reconstruction_error", err_train, "train")
mlca_runtime.record_metric("reconstruction_error", err_val, "val")
mlca_runtime.record_metric("pca_n_components", float(n_comp), "train")

mlca_runtime.save_model(full, "model.pkl")

print(f"features: {len(feat_cols)}, pca components: {n_comp}")
print(f"train reconstruction_error (frontend space): {err_train:.4g}")
print(f"val reconstruction_error (frontend space): {err_val:.4g}")
