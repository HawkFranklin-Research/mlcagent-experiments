import json
import pickle

import numpy as np
import pandas as pd

import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())


def required_key(obj, key):
    if key not in obj:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(obj.keys())}")
    return obj[key]


targets = required_key(protocol, "targets")
excluded = required_key(protocol, "excluded_columns")
task = required_key(protocol, "task")
group_column = protocol.get("group_column", None)

if not isinstance(excluded, (list, tuple, set)):
    raise TypeError(f"excluded_columns must be list-like, got {type(excluded).__name__}")

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_GK4CB9FZ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

reserved = set(targets) | set(excluded)
if group_column is not None:
    reserved.add(group_column)


def is_key_like(series):
    n = len(series)
    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return series.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.nunique(dropna=False) == n
    return False


if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    bad = [c for c in feature_cols if c in reserved]
    if bad:
        raise ValueError(f"Saved preprocessor expects reserved columns: {bad}")
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Saved preprocessor expects missing columns: {missing}")
else:
    candidates = [c for c in train.columns if c not in reserved]
    feature_cols = [c for c in candidates if not is_key_like(train[c])]

if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]
y = train[target]
X = train[feature_cols]

try:
    n_features = len(pre.get_feature_names_out(input_features=feature_cols))
except Exception:
    probe = train.iloc[:1][feature_cols]
    n_features = pre.transform(probe).shape[1]

k = min(20, n_features)
if k < 1:
    raise ValueError("No encoded features available for selection")

selector = SelectKBest(score_func=mutual_info_classif, k=k)
if isinstance(pre, Pipeline):
    if any(isinstance(step, SelectKBest) for _, step in pre.steps):
        raise ValueError("Saved preprocessor already contains SelectKBest")
    pipe = Pipeline(steps=list(pre.steps) + [("selector", selector)])
else:
    pipe = Pipeline(steps=[("preprocessor", pre), ("selector", selector)])

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
selected = int(fitted.named_steps["selector"].get_support().sum())
mlca_runtime.record_metric("encoded_features", n_features, "train")
mlca_runtime.record_metric("selected_features", selected, "train")
mlca_runtime.save_model(fitted, "selector.pkl")
print(f"train rows: {len(train)}")
print(f"raw features: {len(feature_cols)}")
print(f"encoded features: {n_features}")
print(f"k: {k}, selected: {selected}")
print("saved selector.pkl")
