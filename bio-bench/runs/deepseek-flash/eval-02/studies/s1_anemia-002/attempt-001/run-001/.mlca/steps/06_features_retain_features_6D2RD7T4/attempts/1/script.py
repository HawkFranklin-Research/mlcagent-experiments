import json
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

with open(inputs["05_preprocess_fit_preprocessing_3WJD77DF/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

if hasattr(pre, "feature_names_in_"):
    feat_cols = [c for c in pre.feature_names_in_ if c not in drop]
else:
    feat_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.is_unique and (pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        feat_cols.append(c)

X = train[feat_cols]
y = train[targets[0]]
sel = FunctionTransformer()  # importable identity passthrough

if isinstance(pre, Pipeline):
    steps = list(pre.steps) + [("select_all", sel)]
else:
    steps = [("preprocess", pre), ("select_all", sel)]

pipe = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"features={len(feat_cols)} rows={len(X)} target={targets[0]}")
