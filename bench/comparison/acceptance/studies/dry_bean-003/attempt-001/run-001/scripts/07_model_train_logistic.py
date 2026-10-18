import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop]
y = train[targets[0]]

with open(inp["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_KAD36QCW/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None


def contains_preprocessor(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(contains_preprocessor(step) for step in obj.named_steps.values())
    return False


if contains_preprocessor(sel):
    frontend = sel
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

names = input_names(frontend)
if names is None:
    names = input_names(sel) if contains_preprocessor(sel) else input_names(pre)
if names is None:
    names = feature_cols

missing = [c for c in names if c not in train.columns]
if missing:
    raise KeyError(f"missing expected input columns: {missing}")

X = train[names]

model = Pipeline([
    ("frontend", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print("saved model.pkl", "n_train", len(X), "n_features", len(names))
