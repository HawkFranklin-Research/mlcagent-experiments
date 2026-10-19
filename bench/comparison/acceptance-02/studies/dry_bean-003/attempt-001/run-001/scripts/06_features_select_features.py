import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing key {key!r}; available top-level keys: {list(d.keys())}")
    return d[key]


targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")

if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

blocked = set(targets) | set(excluded)
if group_col:
    blocked.add(group_col)
base_cols = [c for c in train.columns if c not in blocked]


def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s) and s.notna().all()
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        vals = np.sort(s.dropna().to_numpy())
        return len(vals) > 1 and np.all(np.diff(vals) == 1) and vals[0] in (0, 1)
    return False


pre_cols = list(getattr(pre, "feature_names_in_", []))
if pre_cols:
    bad = [c for c in pre_cols if c in blocked]
    if bad:
        raise ValueError(f"Preprocessor expects blocked columns: {bad}")
    missing = [c for c in pre_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects missing columns: {missing}")
    feature_cols = pre_cols
else:
    feature_cols = [c for c in base_cols if not is_keylike(train[c])]

if not feature_cols:
    raise ValueError("No feature columns")

X_train = train[feature_cols]
y_train = train[target]

try:
    n_out = pre.transform(X_train).shape[1]
except Exception:
    n_out = len(feature_cols)

k = min(20, n_out)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    last = pre.steps[-1][1]
    if not hasattr(last, "transform"):
        raise ValueError("Saved preprocessor pipeline does not end in a transformer")
    pipe = Pipeline(steps=list(pre.steps) + [("selector", selector)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", selector)])

if n_out <= 300:
    if hasattr(pre, "set_output"):
        pre.set_output(transform="pandas")
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
selected = int(pipe.named_steps["selector"].get_support().sum())

mlca_runtime.record_metric("n_input_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_selected_features", selected, "train")
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"train rows={len(train)} input_features={len(feature_cols)} transformed={n_out} selected={selected}")
