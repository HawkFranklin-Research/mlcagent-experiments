import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import VarianceThreshold

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded) | {group_col}
feature_cols = [c for c in train.columns if c not in drop]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

X = train[feature_cols]

selector = VarianceThreshold(threshold=-1.0)
mlca_runtime.fit_record(selector, X, "train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train shape: {train.shape}")
print(f"feature cols: {len(feature_cols)}")
print(f"selector kept: {int(selector.get_support().sum())}")
