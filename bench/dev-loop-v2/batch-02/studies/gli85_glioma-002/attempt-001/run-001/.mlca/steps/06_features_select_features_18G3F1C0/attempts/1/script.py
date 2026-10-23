import json
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
group = protocol.get("group_column")
target = targets[0]

if target not in train.columns:
    raise KeyError(f"Target {target!r} is not in the training table")

with open(inputs["05_preprocess_fit_preprocessing_M7MKT30F/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)

base = [c for c in train.columns if c not in set(targets) | set(excluded) | ({group} if group else set())]


def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique())
        if len(u) == n:
            return np.array_equal(u, np.arange(u[0], u[0] + n))
    return False


computed_features = [c for c in base if not is_key_like(train[c])]
saved_names = getattr(preproc, "feature_names_in_", None)
feature_cols = list(saved_names) if saved_names is not None else computed_features

bad = set(feature_cols) & (set(targets) | set(excluded) | ({group} if group else set()))
if bad:
    raise ValueError(f"Feature list contains target/excluded/group columns: {sorted(bad)}")
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Feature columns missing from training table: {missing}")

X = train[feature_cols]
y = train[target]
n_out = preproc.transform(X).shape[1]
k = min(20, n_out)
if k < 1:
    raise ValueError("Preprocessing produced no features")

pipe = Pipeline([
    ("preprocessor", preproc),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")
mlca_runtime.record_metric("selected_features", float(k), "train")

print(f"train rows={len(train)} raw_features={len(feature_cols)}")
print(f"encoded_width={n_out} k={k}")
print("saved selector.pkl")
