import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if pd.api.types.is_float_dtype(s):
        return False
    if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)):
        return False
    return len(s) > 0 and s.nunique(dropna=True) == len(s)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

X_train = train[feat_cols]
X_val = val[feat_cols]

with open(inputs["05_preprocess_fit_preprocessing_E2NYE0HC/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_8M4G64VT/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def contains(outer, inner):
    if outer is inner:
        return True
    if isinstance(outer, Pipeline):
        return any(step is inner for step in outer.named_steps.values())
    return False

if contains(sel, pre):
    frontend = sel
elif contains(pre, sel):
    frontend = pre
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

n_features = frontend.transform(X_train.head(5)).shape[1]
n_comp = min(8, n_features)

pipe = Pipeline([("frontend", frontend), ("pca", PCA(n_components=n_comp))])
mlca_runtime.fit_record(pipe, X_train, "train")

Xv = frontend.transform(X_val)
pca = pipe.named_steps["pca"]
Xr = pca.inverse_transform(pca.transform(Xv))
err = float(np.mean((Xv - Xr) ** 2))
mlca_runtime.record_metric("reconstruction_error", err, "val")

mlca_runtime.save_model(pipe, "model.pkl")
print("features:", len(feat_cols), "pca_components:", n_comp)
print("val reconstruction_error (frontend feature space):", round(err, 4))
