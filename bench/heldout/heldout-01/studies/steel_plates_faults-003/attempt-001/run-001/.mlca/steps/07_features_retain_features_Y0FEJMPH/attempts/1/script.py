import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inp["train"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if s.dtype == object and s.nunique(dropna=True) == len(train):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(train):
        v = s.dropna().to_numpy()
        if len(v) > 1 and (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)):
            continue
    feature_cols.append(c)

selector = ColumnTransformer(
    [("pass", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

mlca_runtime.fit_record(selector, train[feature_cols], "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"rows={len(train)} features={len(feature_cols)}")
print("saved selector_all.pkl")
