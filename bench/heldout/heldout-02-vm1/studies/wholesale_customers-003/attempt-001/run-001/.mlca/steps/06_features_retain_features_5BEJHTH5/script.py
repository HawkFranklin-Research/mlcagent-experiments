import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_HBN850GP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Protocol missing '{k}'. Available keys: {sorted(protocol.keys())}")

try:
    feature_cols = list(preprocessor.feature_names_in_)
except AttributeError:
    targets = set(protocol["targets"])
    excluded = set(protocol["excluded_columns"])
    group = protocol["group_column"]
    feature_cols = []
    for c in train.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = train[c]
        nun = s.nunique(dropna=False)
        if s.dtype.kind == "O" and nun == len(s):
            continue
        if s.dtype.kind in "iu" and nun == len(s) and len(s) > 1:
            continue
        feature_cols.append(c)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Feature columns missing from train: {missing}")

try:
    preprocessor.set_output(transform="pandas")
except Exception:
    pass

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", FunctionTransformer()),
])
mlca_runtime.fit_record(pipeline, train[feature_cols], "train")
mlca_runtime.save_model(pipeline, "selector_all.pkl")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
print(f"preprocessor={type(preprocessor).__name__} features={len(feature_cols)} rows={len(train)}")
print("saved selector_all.pkl")
