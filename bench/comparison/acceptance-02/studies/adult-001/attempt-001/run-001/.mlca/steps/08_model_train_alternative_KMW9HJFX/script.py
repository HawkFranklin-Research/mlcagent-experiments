import json
import pickle

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import mlca_runtime

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def required(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d)}")
    return d[key]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
task = required(protocol, "task")
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
y = train[targets[0]]

hgb_kwargs = {"random_state": seed}
if y.nunique() == 2 and y.value_counts(normalize=True).min() < 0.10:
    hgb_kwargs["class_weight"] = "balanced"

prev = pickle.loads(inputs["latest_model"].read_bytes())
if isinstance(prev, Pipeline):
    last_name = prev.steps[-1][0]
    new_pipe = clone(prev)
    new_pipe.set_params(**{last_name: HistGradientBoostingClassifier(**hgb_kwargs)})
else:
    pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_ABA216ED/preprocessor.pkl"].read_bytes())
    sel = pickle.loads(inputs["06_features_select_features_DJ7CCJDF/selector.pkl"].read_bytes())
    new_pipe = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("model", HistGradientBoostingClassifier(**hgb_kwargs)),
    ])

expected = None
for obj in (prev, new_pipe):
    if hasattr(obj, "feature_names_in_"):
        expected = list(obj.feature_names_in_)
        break
if expected is None and isinstance(prev, Pipeline):
    first = prev.steps[0][1]
    if hasattr(first, "feature_names_in_"):
        expected = list(first.feature_names_in_)

bad = set(targets) | set(excluded)
if group is not None:
    bad.add(group)

if expected is None:
    expected = [c for c in train.columns if c not in bad]
else:
    bad_expected = [c for c in expected if c in bad]
    if bad_expected:
        raise ValueError(f"Saved pipeline expects target/excluded/group columns: {bad_expected}")
    expected = [c for c in expected if c not in bad]

missing = [c for c in expected if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

X = train[expected].copy()
fitted = mlca_runtime.fit_record(new_pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"train_rows={len(X)} features={X.shape[1]}")
print(f"classes={list(fitted.classes_)}")
print("saved alternative_model.pkl")
