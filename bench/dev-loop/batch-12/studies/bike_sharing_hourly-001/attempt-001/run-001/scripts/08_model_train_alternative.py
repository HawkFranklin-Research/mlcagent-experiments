import json
import pickle

import pandas as pd

import mlca_runtime
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())
required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
train = pd.read_parquet(ins["train"])


def get_features(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise AttributeError("No feature_names_in_ found")


with open(ins["07_model_train_baseline_VKCYA88G/model.pkl"], "rb") as f:
    base = pickle.load(f)
if not isinstance(base, Pipeline):
    raise RuntimeError(f"Baseline is not a Pipeline: {type(base)}")

expected = get_features(base)
forbidden = set(targets) | set(excluded) | ({group} if group else set())
leak = set(expected) & forbidden
if leak:
    raise ValueError(f"Target/excluded/group columns in features: {leak}")
missing_cols = set(expected) - set(train.columns)
if missing_cols:
    raise KeyError(f"Missing feature columns: {missing_cols}")

steps = list(base.steps[:-1])
rf = RandomForestRegressor(n_estimators=100, random_state=42)
new_pipe = Pipeline(steps + [("model", rf)])
mlca_runtime.fit_record(new_pipe, train[expected], "train", y=train[targets])
mlca_runtime.save_model(new_pipe, "alternative_model.pkl")
print(f"train rows {len(train)}, features {len(expected)}, targets {targets}")
print("saved alternative_model.pkl")
