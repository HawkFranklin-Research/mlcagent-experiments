import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key}; available keys: {list(proto.keys())}")

targets = proto["targets"] or []
excluded = proto["excluded_columns"] or []
group_col = proto.get("group_column")
task = proto["task"] or {}
metric_name = task.get("primary_metric", "reconstruction_error")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def is_key_like(s: pd.Series) -> bool:
    if s.isna().any():
        return False
    n = len(s)
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique() == n
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique() == n
    return False

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
feature_cols = [
    c for c in train.columns
    if c not in exclude and not is_key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like detection")

X_train = train[feature_cols]
X_val = val[feature_cols]
n_features = len(feature_cols)
n_components = min(8, n_features)

imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
scaler = StandardScaler().set_output(transform="pandas")
pca = PCA(n_components=n_components, random_state=42)
pipeline = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
    ("pca", pca),
])
pipeline = mlca_runtime.fit_record(pipeline, X_train, "train")

frontend = Pipeline(pipeline.steps[:-1])
X_val_front = frontend.transform(X_val)
pca_fitted = pipeline.named_steps["pca"]
X_val_recon = pca_fitted.inverse_transform(pca_fitted.transform(X_val_front))
X_val_front_arr = X_val_front.values if hasattr(X_val_front, "values") else np.asarray(X_val_front)
val_mse = float(mean_squared_error(X_val_front_arr, X_val_recon))

mlca_runtime.record_metric(metric_name, val_mse, "val")
mlca_runtime.record_metric("pca_n_components", float(n_components), "train")
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"features={n_features} train={X_train.shape} val={X_val.shape}")
print(f"n_components={n_components} val_reconstruction_error(pre-PCA space)={val_mse:.6f}")
