import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(
        f"Missing protocol keys {missing}; available top-level keys: "
        f"{sorted(protocol.keys())}"
    )

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
metric_name = protocol["task"].get("primary_metric", "reconstruction_error")

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

candidate_cols = [c for c in train_df.columns if c not in drop]
feature_cols = []
for c in candidate_cols:
    s = train_df[c]
    is_text_key = (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    ) and s.nunique(dropna=False) == len(s)
    is_int_key = (
        pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s)
    )
    if is_text_key or is_int_key:
        continue
    feature_cols.append(c)

if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions")

n_features = len(feature_cols)
k = min(8, n_features)

pre = ColumnTransformer(
    [("keep", "passthrough", feature_cols)],
    remainder="drop",
)
pre.set_output(transform="pandas")

model = Pipeline(
    [
        ("select", pre),
        ("pca", PCA(n_components=k, random_state=42)),
    ]
)
model.set_output(transform="pandas")

X_train = train_df[feature_cols].copy()
model = mlca_runtime.fit_record(model, X_train, partition="train")
pca = model.named_steps["pca"]

def recon_mse(X):
    Xt = model.transform(X)
    Xr = pca.inverse_transform(Xt)
    Xv = X[feature_cols].to_numpy(dtype=float)
    return float(np.mean((Xv - Xr) ** 2))

train_mse = recon_mse(X_train)
mlca_runtime.record_metric(metric_name, train_mse, split="train")

if all(c in val_df.columns for c in feature_cols):
    X_val = val_df[feature_cols].copy()
    val_mse = recon_mse(X_val)
    mlca_runtime.record_metric(metric_name, val_mse, split="val")
else:
    val_mse = float("nan")

mlca_runtime.save_model(model, "model.pkl")

print(f"features={n_features} components={k}")
print(f"train_reconstruction_error={train_mse:.6g}")
print(f"val_reconstruction_error={val_mse:.6g}")
