import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError("This action expects exactly one target")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if (
        pd.api.types.is_integer_dtype(s)
        and s.notna().all()
        and len(train) > 2
        and s.nunique(dropna=True) == len(train)
    ):
        continue
    feature_cols.append(col)
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X = train.loc[:, feature_cols]
y = train[targets[0]]

expected = getattr(pre, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    if any(col in drop for col in expected):
        raise ValueError("Saved preprocessor expects a target or excluded column")
    missing = set(expected) - set(X.columns)
    if missing:
        raise ValueError(f"Saved preprocessor expects missing columns: {sorted(missing)}")
    X = X.loc[:, expected]

n_features = pre.transform(X).shape[1]
k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = list(pre.steps) + [("selectkbest", selector)]
else:
    steps = [("preprocessing", pre), ("selectkbest", selector)]

pipe = Pipeline(steps)
if n_features <= 300:
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")
print("selector.pkl saved")
