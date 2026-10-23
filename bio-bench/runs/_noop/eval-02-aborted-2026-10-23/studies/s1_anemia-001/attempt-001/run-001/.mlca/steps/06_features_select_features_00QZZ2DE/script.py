import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())
available = sorted(protocol)

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r} in _protocol; available top-level keys: {available}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"] or []
group = protocol.get("group_column")
target = targets[0]

if target not in train.columns:
    raise KeyError(f"Target {target!r} is not present in the training table")

with open(inputs["05_preprocess_fit_preprocessing_PTKJZF9S/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if not hasattr(pre, "feature_names_in_"):
    raise AttributeError("Loaded preprocessor has no feature_names_in_")

feature_cols = list(pre.feature_names_in_)
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Feature columns missing from train: {missing}")

bad = [
    c for c in feature_cols
    if c == target or c in excluded or (group is not None and c == group)
]
if bad:
    raise ValueError(f"Target/excluded/group columns reached the preprocessor input: {bad}")

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    n_out = pre.transform(train[feature_cols].iloc[:1]).shape[1]

k = min(20, n_out)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = list(pre.steps)
    names = {name for name, _ in steps}
    new_name = "selectkbest"
    while new_name in names:
        new_name = "_" + new_name
    pipeline = Pipeline(steps + [(new_name, selector)])
else:
    pipeline = Pipeline([("preprocessor", pre), ("selectkbest", selector)])

pipeline = mlca_runtime.fit_record(
    pipeline, train[feature_cols], "train", y=train[target]
)
mlca_runtime.save_model(pipeline, "selector.pkl")

print("train rows:", len(train))
print("raw feature columns:", len(feature_cols))
print("preprocessed feature count:", n_out)
print("selected k:", k)
