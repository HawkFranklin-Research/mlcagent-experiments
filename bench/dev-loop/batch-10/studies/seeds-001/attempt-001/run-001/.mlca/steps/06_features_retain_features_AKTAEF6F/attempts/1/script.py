import json
import mlca_runtime
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

needed = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol)}")

targets = protocol["targets"] or []
excluded = protocol["excluded_columns"] or []
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

candidates = [c for c in train.columns if c not in drop]
feature_cols = []
for c in candidates:
    s = train[c]
    is_text = s.dtype == object or pd.api.types.is_string_dtype(s)
    is_int = pd.api.types.is_integer_dtype(s)
    all_unique = s.notna().all() and s.nunique(dropna=True) == len(s)
    if (is_text or is_int) and all_unique:
        continue
    feature_cols.append(c)

X = train[feature_cols]
selector = FunctionTransformer(func=None, validate=False)
fitted = mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"fitted passthrough selector on {X.shape[0]} rows, {X.shape[1]} features")
