import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(proto.keys())}")

targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group = proto["group_column"]
task = proto["task"]

train = pd.read_parquet(inputs["train"])
with open(inputs["06_preprocess_fit_preprocessing_VD4Z4Z0N/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feature_cols = getattr(pre, "feature_names_in_", None)
if feature_cols is None:
    feature_cols = [
        c for c in train.columns
        if c not in set(targets) and c not in excluded and c != group
    ]
else:
    feature_cols = list(feature_cols)

X = train[feature_cols]
y = train[targets[0]]

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    try:
        n_out = pre.transform(X.head(1)).shape[1]
    except Exception:
        n_out = len(feature_cols)

k = max(1, min(20, n_out))
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = [(name, step) for name, step in pre.steps if name != "selector"]
    steps.append(("selector", selector))
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", selector)])

pipe = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"train rows: {len(train)}")
print(f"features in/out: {len(feature_cols)}/{n_out}, k={k}")
print("saved selector.pkl")
