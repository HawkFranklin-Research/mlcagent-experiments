import mlca_runtime
import json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(train):
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(train):
            feature_cols.remove(c)

X = train[feature_cols]
selector = FunctionTransformer(validate=False)
mlca_runtime.fit_record(selector, X, "train", y=None)
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train shape: {train.shape}")
print(f"features: {len(feature_cols)}")
print("saved selector_all.pkl")
