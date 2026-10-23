import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
if "group_column" not in protocol:
    raise KeyError(f"group_column missing; available keys: {sorted(protocol)}")
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_M7AXCKT1/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_XFCK1C59/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target_col = targets[0]


def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    return None


raw_cols = feature_names(pre)
if raw_cols is None:
    excluded_set = set(targets) | set(excluded)
    candidates = [c for c in train.columns if c not in excluded_set and c != group_col]

    def is_keylike(s):
        if s.nunique(dropna=True) != len(s):
            return False
        if pd.api.types.is_integer_dtype(s):
            return True
        return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

    raw_cols = [c for c in candidates if not is_keylike(train[c])]

missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"missing feature columns: {missing[:5]}")

X_train = train[raw_cols].copy()
y_train = train[target_col]

sel_cols = feature_names(sel)
same_raw = (
    sel_cols is not None
    and len(sel_cols) == len(raw_cols)
    and all(a == b for a, b in zip(sel_cols, raw_cols))
)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel is None:
    model = Pipeline([("preprocess", pre), ("model", clf)])
    X_fit = X_train
elif isinstance(sel, Pipeline) and same_raw:
    model = Pipeline([("selector", sel), ("model", clf)])
    X_fit = train[list(sel.feature_names_in_)].copy()
else:
    model = Pipeline([("preprocess", pre), ("selector", sel), ("model", clf)])
    X_fit = X_train

fitted = mlca_runtime.fit_record(model, X_fit, "train", y_train)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows={len(train)} raw features={len(raw_cols)} target={target_col}")
print(f"pipeline steps={list(fitted.named_steps)}")
print(f"classes={list(fitted.named_steps['model'].classes_)}")
