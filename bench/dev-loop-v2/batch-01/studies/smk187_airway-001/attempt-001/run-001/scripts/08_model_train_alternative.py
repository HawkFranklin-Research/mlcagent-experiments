import json
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import mlca_runtime

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["latest_model"], "rb") as f:
    prev = pickle.load(f)

if hasattr(prev, "feature_names_in_"):
    raw_cols = list(prev.feature_names_in_)
else:
    excluded = set(protocol.get("excluded_columns", [])) | set(targets)
    group_col = protocol.get("group_column")
    if group_col:
        excluded.add(group_col)
    raw_cols = [c for c in train.columns if c not in excluded]

X = train[raw_cols].copy()
y = train[target]

kwargs = {"random_state": seed}
counts = y.value_counts(normalize=True)
if len(counts) > 1 and counts.min() < 0.10:
    kwargs["class_weight"] = "balanced"
clf = HistGradientBoostingClassifier(**kwargs)

if isinstance(prev, Pipeline):
    pipe = Pipeline(prev.steps[:-1] + [("alternative_model", clf)])
else:
    with open(inputs["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_NTK41RPZ/selector.pkl"], "rb") as f:
        sel = pickle.load(f)

    if isinstance(sel, Pipeline) and "preprocessor" in sel.named_steps:
        pipe = Pipeline(sel.steps[:-1] + [("alternative_model", clf)])
    else:
        pipe = Pipeline([
            ("preprocessor", pre),
            ("selector", sel),
            ("alternative_model", clf),
        ])

    if hasattr(pre, "feature_names_in_"):
        raw_cols = list(pre.feature_names_in_)
        X = train[raw_cols].copy()

mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print("saved alternative_model.pkl")
