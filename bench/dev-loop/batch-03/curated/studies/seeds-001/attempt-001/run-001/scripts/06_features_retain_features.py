import mlca_runtime
import json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])

feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or col == group:
        continue
    if train[col].dtype == "object" or train[col].dtype.name == "string":
        if train[col].nunique() == len(train):
            continue
    feature_cols.append(col)

X = train[feature_cols].copy()

selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Fitted passthrough selector on {X.shape[0]} rows, {X.shape[1]} features.")
