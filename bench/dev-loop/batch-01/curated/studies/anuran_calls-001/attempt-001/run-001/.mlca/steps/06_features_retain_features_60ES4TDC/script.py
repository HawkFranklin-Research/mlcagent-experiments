import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

for key in ["targets", "excluded_columns", "group_column"]:
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]

feature_cols = [c for c in train.columns if c not in targets + excluded + [group]]
for c in feature_cols[:]:
    if train[c].dtype == "object" or pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            feature_cols.remove(c)

X_train = train[feature_cols].copy()

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", feature_cols)],
    remainder="drop"
)
selector.set_output(transform="pandas")

selector = mlca_runtime.fit_record(selector, X_train, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Train shape: {train.shape}")
print(f"Features: {len(feature_cols)}")
