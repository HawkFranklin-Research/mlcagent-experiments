import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
target_col = targets[0]
y = train[target_col]

with open(inputs["latest_model"], "rb") as f:
    base = pickle.load(f)

if hasattr(base, "feature_names_in_"):
    feature_cols = list(base.feature_names_in_)
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group_col]

counts = pd.Series(y).value_counts()
minority_frac = counts.min() / counts.sum()

est = HistGradientBoostingClassifier(
    random_state=42,
    class_weight=("balanced" if minority_frac < 0.10 else None),
)

if isinstance(base, Pipeline):
    steps = list(base.steps[:-1]) + [("model", est)]
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([("model", est)])

X = train[feature_cols]
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"rows={X.shape[0]} features={X.shape[1]} minority_frac={minority_frac:.3f}")
print(f"steps={list(pipe.named_steps.keys())}")
