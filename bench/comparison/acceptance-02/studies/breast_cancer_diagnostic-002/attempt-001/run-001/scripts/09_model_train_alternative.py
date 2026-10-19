import mlca_runtime
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing _protocol key {key!r}; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column", None)
target = targets[0]

train = pd.read_parquet(inputs["train"])
y = train[target]

def first_input_features(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            fn = getattr(step, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
    return None

with open(inputs["08_model_train_logistic_71AKM31B/model.pkl"], "rb") as f:
    base = pickle.load(f)

if isinstance(base, Pipeline):
    steps = list(base.steps)
    steps[-1] = ("histgradientboostingclassifier",
                 HistGradientBoostingClassifier(random_state=seed))
    pipeline = Pipeline(steps)
    input_cols = first_input_features(base)
else:
    with open(inputs["06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["07_features_select_features_B35TTT17/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipeline = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("histgradientboostingclassifier",
         HistGradientBoostingClassifier(random_state=seed)),
    ])
    input_cols = first_input_features(pre)

if input_cols is None:
    input_cols = [c for c in train.columns
                  if c not in targets and c not in excluded and c != group]
else:
    bad = [c for c in input_cols if c in targets or c in excluded or c == group]
    if bad:
        raise ValueError(f"Saved pipeline expects target/excluded/group columns: {bad}")
    missing = [c for c in input_cols if c not in train.columns]
    if missing:
        raise KeyError(f"Saved pipeline expects columns missing from train: {missing}")

X = train.loc[:, input_cols]
mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")
print("alternative_model.pkl saved", X.shape)
