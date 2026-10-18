import json
import pickle
from functools import partial

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def need(obj, key, context):
    if key not in obj:
        raise KeyError(f"Missing {key!r} in {context}; available keys: {sorted(obj)}")
    return obj[key]


targets = need(protocol, "targets", "_protocol")
excluded = need(protocol, "excluded_columns", "_protocol")
group = need(protocol, "group_column", "_protocol")
task = need(protocol, "task", "_protocol")
need(task, "type", "_protocol['task']")

if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"Target {target!r} not found in train")

y = train[target]
with open(inputs["06_preprocess_fit_preprocessing_4H3435T3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

forbidden = set(targets) | set(excluded) | {group}
if hasattr(pre, "feature_names_in_"):
    input_cols = list(pre.feature_names_in_)
else:
    manifest = json.loads(
        inputs["06_preprocess_fit_preprocessing_4H3435T3/manifest.json"].read_text()
    )
    input_cols = None
    for key in ("feature_columns", "input_columns", "features", "columns"):
        if key in manifest and isinstance(manifest[key], list):
            input_cols = list(manifest[key])
            break
    if input_cols is None:
        input_cols = []
        for col in train.columns:
            if col in forbidden:
                continue
            s = train[col]
            if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
                continue
            input_cols.append(col)

bad = forbidden.intersection(input_cols)
if bad:
    raise ValueError(f"Forbidden columns in input schema: {sorted(bad)}")
missing = [c for c in input_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing input columns in train: {missing}")

X_train = train[input_cols].copy()
n_features = pre.transform(X_train).shape[1]
k = min(20, n_features)

select_step = SelectKBest(
    score_func=partial(mutual_info_classif, random_state=42),
    k=k,
)

if isinstance(pre, Pipeline):
    step_name = "selectkbest"
    if step_name in pre.named_steps:
        step_name = "select_k_best"
    final = Pipeline(pre.steps + [(step_name, select_step)])
else:
    step_name = "selectkbest"
    final = Pipeline([("preprocessor", pre), (step_name, select_step)])

mlca_runtime.fit_record(final, X_train, partition="train", y=y)
mlca_runtime.save_model(final, "selector.pkl")

n_selected = int(final.named_steps[step_name].get_support().sum())
mlca_runtime.record_metric("selector_n_features", int(n_features), "train")
mlca_runtime.record_metric("selector_k", int(k), "train")
mlca_runtime.record_metric("selector_n_selected", n_selected, "train")

print(f"train rows: {len(train)}")
print(f"input features: {len(input_cols)}")
print(f"preprocessed features: {n_features}")
print(f"selected features: {n_selected} (k={k})")
