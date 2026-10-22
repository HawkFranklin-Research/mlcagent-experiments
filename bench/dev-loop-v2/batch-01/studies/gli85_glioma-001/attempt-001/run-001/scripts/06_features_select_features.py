import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inp["train"])
pre = pickle.load(open(inp["05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl"], "rb"))
json.loads(inp["05_preprocess_fit_preprocessing_S343E7S5/manifest.json"].read_text())

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

cand = [c for c in train.columns if c not in drop]
key_like = []
n_rows = len(train)
for c in cand:
    s = train[c]
    is_text = pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and s.nunique(dropna=False) >= n_rows:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique(dropna=False) >= n_rows:
        vals = np.sort(s.to_numpy())
        if len(vals) == n_rows and np.all(np.diff(vals) == 1) and vals[0] in (0, 1):
            key_like.append(c)

cand = [c for c in cand if c not in set(key_like)]

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

names = input_names(pre)
if names is not None:
    bad = [c for c in names if c not in train.columns or c in drop]
    if bad:
        raise ValueError(f"Saved preprocessor expects unavailable/forbidden columns: {bad[:5]}")
    feature_columns = names
else:
    feature_columns = cand

if set(feature_columns) & set(targets):
    raise ValueError("Target columns must not reach the feature selector")

X_train = train[feature_columns].copy()
y_train = train[targets[0]]

try:
    n_out = pre.transform(X_train).shape[1]
except Exception:
    n_out = len(feature_columns)
k = min(20, int(n_out))

has_selector = isinstance(pre, Pipeline) and any(isinstance(step, SelectKBest) for _, step in pre.steps)
if has_selector:
    model = pre
    for name, step in getattr(pre, "steps", []):
        if isinstance(step, SelectKBest):
            model.set_params(**{f"{name}__k": k})
else:
    model = Pipeline([
        ("preprocessor", pre),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
    ])

model = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "selector.pkl")

print(f"train shape={train.shape}; feature columns={len(feature_columns)}")
print(f"SelectKBest k={k}")
