import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise ValueError(f"_protocol missing '{k}'; available keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(s):
    n = s.nunique(dropna=True)
    if n <= 1:
        return False
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return n == len(s)
    return False

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

feature_cols = [c for c in train.columns if c not in drop_cols and not is_key_like(train[c])]
non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c].dtype)]
if non_numeric:
    raise ValueError(f"Unexpected non-numeric features: {non_numeric}")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = X_train.copy()  # reconstruction target = the same features

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
scaler_x = StandardScaler().set_output(transform="pandas")
scaler_y = StandardScaler()
inner = Pipeline([("scaler", scaler_x), ("mlp", mlp)])
model = TransformedTargetRegressor(regressor=inner, transformer=scaler_y)

mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "alternative_model.pkl")

# Reconstruction error in the frontend (scaled) feature space.
fx = model.regressor_.named_steps["scaler"]
true_v = fx.transform(X_val)
pred_v = fx.transform(pd.DataFrame(model.predict(X_val), columns=feature_cols, index=X_val.index))
mse_val = float(np.mean((true_v.values - pred_v.values) ** 2))
mlca_runtime.record_metric("reconstruction_error", mse_val, "val")

true_t = fx.transform(X_train)
pred_t = fx.transform(pd.DataFrame(model.predict(X_train), columns=feature_cols, index=X_train.index))
mse_tr = float(np.mean((true_t.values - pred_t.values) ** 2))
mlca_runtime.record_metric("reconstruction_error", mse_tr, "train")

embedding_dim = int(mlp.hidden_layer_sizes[1])
summary = {
    "n_features": len(feature_cols),
    "embedding_dim": embedding_dim,
    "hidden_layer_sizes": list(mlp.hidden_layer_sizes),
    "reconstruction_error_val": round(mse_val, 6),
    "reconstruction_error_train": round(mse_tr, 6),
    "feature_space": "scaled (StandardScaler fit on train rows)",
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
}
mlca_runtime.save_json(summary, "alternative_model_summary.json", kind="report")
print("features:", len(feature_cols), "| embedding:", embedding_dim)
print("val reconstruction_error (scaled):", round(mse_val, 4))
