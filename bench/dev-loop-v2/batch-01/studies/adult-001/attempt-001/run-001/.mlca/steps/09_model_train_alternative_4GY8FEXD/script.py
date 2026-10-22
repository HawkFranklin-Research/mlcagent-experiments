import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol["group_column"]
    _ = protocol["task"]["primary_metric"]
except KeyError as exc:
    raise RuntimeError(
        f"Missing protocol key {exc}; available top-level keys: {list(protocol)}"
    ) from exc

train = pd.read_parquet(inputs["train"])
target = targets[0]
if target not in train.columns:
    raise ValueError(f"Target column {target!r} is not in train")

loaded = pickle.load(open(inputs["latest_model"], "rb"))
if isinstance(loaded, dict) and "model" in loaded:
    loaded = loaded["model"]

if isinstance(loaded, Pipeline):
    steps = list(loaded.steps[:-1])
    last_name = loaded.steps[-1][0]
    source_for_names = loaded
elif hasattr(loaded, "steps"):
    steps = list(loaded.steps[:-1])
    last_name = loaded.steps[-1][0]
    source_for_names = loaded
else:
    pre = pickle.load(
        open(inputs["06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl"], "rb")
    )
    sel = pickle.load(
        open(inputs["07_features_select_features_Y03WXMRF/selector.pkl"], "rb")
    )
    steps = []
    if isinstance(pre, Pipeline):
        steps.extend(pre.steps)
    else:
        steps.append(("preprocessor", pre))
    if isinstance(sel, Pipeline):
        steps.extend(sel.steps)
    else:
        steps.append(("selector", sel))
    last_name = "model"
    source_for_names = pre

feature_cols = getattr(source_for_names, "feature_names_in_", None)
if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
    n_rows = len(train)
    kept = []
    for col in feature_cols:
        s = train[col]
        is_text = pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)
        is_int = pd.api.types.is_integer_dtype(s.dtype)
        if (is_text or is_int) and n_rows > 1 and s.nunique(dropna=False) == n_rows:
            continue
        kept.append(col)
    feature_cols = kept
else:
    feature_cols = list(feature_cols)

forbidden = set(feature_cols) & (set(targets) | set(excluded))
if group is not None:
    forbidden.add(group)
if forbidden:
    raise ValueError(f"Feature columns include forbidden columns: {sorted(forbidden)}")
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

classifier = HistGradientBoostingClassifier(random_state=seed)
pipe = Pipeline(steps + [(last_name, classifier)])

X = train[feature_cols]
y = train[target]
mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"train rows={len(train)} features={len(feature_cols)}")
print(f"saved pipeline with HistGradientBoostingClassifier(random_state={seed})")
