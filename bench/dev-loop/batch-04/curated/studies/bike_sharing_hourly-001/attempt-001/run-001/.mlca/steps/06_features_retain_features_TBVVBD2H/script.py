import json
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
for key in required:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inputs["train"])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

for c in list(feature_cols):
    s = train[c]
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)

X = train[feature_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

selector = SelectKBest(k='all')
mlca_runtime.fit_record(selector, X, partition='train', y=y)
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Fitted passthrough selector on {X.shape[0]} rows, {X.shape[1]} features.")
