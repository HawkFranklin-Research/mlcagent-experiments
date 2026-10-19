import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"]) if "val" in inputs else None

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
raw_feature_cols = [c for c in train.columns if c not in drop]
for c in list(raw_feature_cols):
    s = train[c]
    if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)):
        if len(s) > 1 and s.nunique(dropna=True) == len(s):
            raw_feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if len(s) > 1 and s.nunique(dropna=True) == len(s):
            raw_feature_cols.remove(c)

with open(inputs["05_preprocess_fit_preprocessing_EJN3HZZN/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_AY9YXAGS/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

for obj in (pre, sel):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            cols = input_cols(step)
            if cols:
                return cols
    return []

def accepts(obj, cols):
    ic = input_cols(obj)
    return bool(ic) and set(ic).issubset(set(cols))

raw_set = set(raw_feature_cols)
if accepts(sel, raw_set):
    frontend = sel
elif accepts(pre, raw_set):
    if hasattr(pre, "get_feature_names_out"):
        pre_out = set(pre.get_feature_names_out())
        sel_in = set(input_cols(sel))
        if sel_in and sel_in.issubset(pre_out):
            frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
        else:
            frontend = pre
    else:
        frontend = pre
else:
    frontend = pre

frontend_cols = input_cols(frontend) or raw_feature_cols
bad = set(frontend_cols) & drop
if bad:
    raise ValueError(f"Frontend input includes excluded/target/group columns: {sorted(bad)}")
missing = set(frontend_cols) - set(train.columns)
if missing:
    raise ValueError(f"Frontend input columns missing from training table: {sorted(missing)}")

X_train = train[frontend_cols].copy()
X_val = val[frontend_cols].copy() if val is not None else None

n_feat = int(frontend.transform(X_train.iloc[:5]).shape[1])
if n_feat <= 300:
    if hasattr(frontend, "set_output"):
        try:
            frontend.set_output(transform="pandas")
        except Exception:
            pass

pca = PCA(n_components=min(8, n_feat), random_state=42)
pipe = Pipeline([("frontend", frontend), ("pca", pca)])
model = mlca_runtime.fit_record(pipe, X_train, "train")

frontend_fitted = model.named_steps["frontend"]
pca_fitted = model.named_steps["pca"]

def dense(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return np.asarray(a, dtype=float)

def reconstruction_error(frame):
    Z = dense(frontend_fitted.transform(frame))
    Z_rec = pca_fitted.inverse_transform(pca_fitted.transform(Z))
    return float(np.mean((Z - dense(Z_rec)) ** 2))

metric_name = protocol["task"].get("primary_metric", "reconstruction_error")
train_err = reconstruction_error(X_train)
mlca_runtime.record_metric(metric_name, train_err, "train")
mlca_runtime.record_metric("embedding_dimension", float(pca_fitted.n_components_), "train")
if X_val is not None:
    val_err = reconstruction_error(X_val)
    mlca_runtime.record_metric(metric_name, val_err, "val")

mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)} input_features={len(frontend_cols)} pca_components={pca_fitted.n_components_}")
print(f"train reconstruction_error={train_err:.4f}")
if X_val is not None:
    print(f"val reconstruction_error={val_err:.4f}")
print("saved model.pkl")
