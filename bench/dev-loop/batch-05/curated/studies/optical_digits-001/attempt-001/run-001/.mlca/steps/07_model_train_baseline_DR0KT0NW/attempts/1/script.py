import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
metric_name = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    return False

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_M7FAZM1F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_271P20QZ/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

if isinstance(sel, Pipeline):
    frontend = sel
    cols = input_cols(sel) or feat_cols
elif isinstance(pre, Pipeline):
    frontend = Pipeline(list(pre.steps) + [("selector", sel)])
    cols = input_cols(pre) or feat_cols
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
    cols = input_cols(pre) or feat_cols

n_comp = min(8, len(feat_cols))
pipe = Pipeline(list(frontend.steps) + [("pca", PCA(n_components=n_comp, random_state=42))])

X = train[cols]
mlca_runtime.fit_record(pipe, X, "train")
mlca_runtime.save_model(pipe, "model.pkl")

pre_sel = Pipeline(pipe.steps[:-1])
X_scaled = pre_sel.transform(X)
pca = pipe.named_steps["pca"]
X_pca = pca.transform(X_scaled)
X_recon = pca.inverse_transform(X_pca)
Xs = X_scaled.values if hasattr(X_scaled, "values") else np.asarray(X_scaled)
Xr = X_recon.values if hasattr(X_recon, "values") else np.asarray(X_recon)
err = float(np.mean((Xs - Xr) ** 2))
mlca_runtime.record_metric(metric_name, err, "train")

print(f"features={len(feat_cols)} components={n_comp} train_err={err:.4f}")
