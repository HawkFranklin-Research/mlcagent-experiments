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
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d)}")
    return d[key]


targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
task = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
missing_targets = [c for c in targets if c not in train.columns]
if missing_targets:
    raise KeyError(f"Missing target columns: {missing_targets}")

if group in (None, ""):
    group = None
elif group not in train.columns:
    raise KeyError(f"Missing group column: {group!r}")


def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().sort_values().reset_index(drop=True)
        return (
            len(vals) == len(s)
            and len(vals) > 1
            and vals.is_unique
            and (vals.to_numpy() == np.arange(len(vals))).all()
        )
    return False


drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = [
    c for c in train.columns
    if c not in drop and not key_like(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_6Y57Q816/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    bad = set(expected) & drop
    if bad:
        raise ValueError(f"Preprocessor expects target/excluded/group columns: {sorted(bad)}")
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing columns: {missing}")
    feature_cols = expected

X = train[feature_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

n_features = int(pre.transform(X).shape[1])
k = min(20, n_features)

pipe = Pipeline([
    ("preprocessor", pre),
    ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
])

fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "selector.pkl")
print("selector fitted")
