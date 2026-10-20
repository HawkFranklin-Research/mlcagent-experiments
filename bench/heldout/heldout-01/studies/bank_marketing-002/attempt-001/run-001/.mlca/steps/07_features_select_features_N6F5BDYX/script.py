import json
import pickle

import pandas as pd

import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
_ = json.loads(paths["06_preprocess_fit_preprocessing_STWKFGVD/manifest.json"].read_text())
with open(paths["06_preprocess_fit_preprocessing_STWKFGVD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

n_rows = len(train)


def keylike(s):
    if s.nunique(dropna=False) != n_rows:
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )


base_features = []
for c in train.columns:
    if c in targets or c in excluded or (group is not None and c == group):
        continue
    if keylike(train[c]):
        continue
    base_features.append(c)

if hasattr(pre, "feature_names_in_") and pre.feature_names_in_ is not None:
    names = list(pre.feature_names_in_)
    bad = [c for c in names if c in targets or c in excluded or (group is not None and c == group)]
    if bad:
        raise ValueError(f"Saved preprocessor expects excluded/target columns: {bad}")
    missing = [c for c in names if c not in train.columns]
    if missing:
        raise KeyError(f"Missing columns for preprocessor: {missing}")
    feature_cols = names
else:
    feature_cols = base_features

if not feature_cols:
    raise ValueError("No feature columns available")

k = min(20, len(feature_cols))
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = [(n, o) for n, o in pre.steps if n != "selectkbest"]
    pipe = Pipeline(steps + [("selectkbest", selector)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selectkbest", selector)])

try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

mlca_runtime.fit_record(
    pipe,
    train[feature_cols],
    partition="train",
    y=train[targets[0]],
)
mlca_runtime.save_model(pipe, "selector.pkl")
print("train_rows", n_rows, "feature_cols", len(feature_cols), "k", k)
