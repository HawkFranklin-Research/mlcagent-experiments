import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import clone

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
for key in required:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
task = protocol["task"]
primary_metric = task["primary_metric"]

with open(inp["06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

# Build feature columns: all except targets, excluded, group, and key-like columns
all_cols = list(train.columns)
exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)

def is_key_like(s):
    if s.isna().any():
        return False
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s) and s.min() >= 0
    if s.dtype == object:
        return s.nunique() == len(s)
    return False

feature_cols = [c for c in all_cols if c not in exclude and not is_key_like(train[c])]
X_train = train[feature_cols]
y_train = train[targets[0]]

# Build pipeline: preprocessor + SelectKBest
if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
    steps.append(('selector', SelectKBest(score_func=mutual_info_classif)))
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_classif))
    ])

# Determine n_features after preprocessing
preproc_fitted = clone(preprocessor)
X_trans = preproc_fitted.fit_transform(X_train, y_train) if not isinstance(preprocessor, Pipeline) else preproc_fitted.fit(X_train, y_train).transform(X_train)
n_features = X_trans.shape[1] if hasattr(X_trans, 'shape') else len(X_trans[0])
k = min(20, n_features)
print(f"n_features after preprocessing: {n_features}, k={k}")

pipeline.set_params(selector__k=k)
mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")
print("Saved selector.pkl")
