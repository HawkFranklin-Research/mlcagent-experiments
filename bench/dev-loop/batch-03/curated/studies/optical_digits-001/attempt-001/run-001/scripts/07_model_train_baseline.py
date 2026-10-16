import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}. Available: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(s):
    if s.dtype.kind in 'OSU':
        return s.nunique(dropna=False) == len(s)
    if s.dtype.kind in 'iu':
        return s.nunique(dropna=False) == len(s)
    return False

all_cols = list(train.columns)
key_like = [c for c in all_cols if is_key_like(train[c])]
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col and c not in key_like]

with open(inputs["05_preprocess_fit_preprocessing_NFXS64E4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_HGAM38DP/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def get_input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

if isinstance(selector, Pipeline):
    frontend = selector
    expected_order = get_input_names(selector)
elif preprocessor is not None:
    frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    expected_order = get_input_names(preprocessor)
else:
    frontend = selector
    expected_order = get_input_names(selector)

if expected_order is not None:
    if set(expected_order) != set(feature_cols):
        raise ValueError("Saved frontend expects different columns than feature contract.")
    feature_cols = expected_order

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

n_features = len(feature_cols)
n_components = min(8, n_features)
pipeline = Pipeline([
    ("frontend", frontend),
    ("pca", PCA(n_components=n_components, random_state=42))
])

fitted = mlca_runtime.fit_record(pipeline, X_train, partition="train")
mlca_runtime.save_model(fitted, "model.pkl")

frontend_fitted = fitted.named_steps["frontend"]
pca_fitted = fitted.named_steps["pca"]

Z_train = frontend_fitted.transform(X_train)
Z_val = frontend_fitted.transform(X_val)
Z_train_rec = pca_fitted.inverse_transform(pca_fitted.transform(Z_train))
Z_val_rec = pca_fitted.inverse_transform(pca_fitted.transform(Z_val))

mse_train = float(np.mean((Z_train - Z_train_rec) ** 2))
mse_val = float(np.mean((Z_val - Z_val_rec) ** 2))

mlca_runtime.record_metric("reconstruction_error", mse_train, split="train")
mlca_runtime.record_metric("reconstruction_error", mse_val, split="val")

print(f"features: {len(feature_cols)}")
print(f"pca components: {n_components}")
print(f"train reconstruction_error: {mse_train:.4g}")
print(f"val reconstruction_error: {mse_val:.4g}")
