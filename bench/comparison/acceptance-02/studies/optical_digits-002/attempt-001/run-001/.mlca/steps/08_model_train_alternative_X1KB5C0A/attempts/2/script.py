import json
import mlca_runtime
import pandas as pd
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets") or []
excluded = protocol.get("excluded_columns") or []
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]

X = train[feature_cols].copy()
y = train[feature_cols].copy()

est = MLPRegressor(
    hidden_layer_sizes=(16, 8, 16),
    max_iter=300,
    random_state=42,
)
fitted = mlca_runtime.fit_record(est, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print("train shape:", train.shape)
print("n_features:", len(feature_cols))
print("bottleneck width: 8")
