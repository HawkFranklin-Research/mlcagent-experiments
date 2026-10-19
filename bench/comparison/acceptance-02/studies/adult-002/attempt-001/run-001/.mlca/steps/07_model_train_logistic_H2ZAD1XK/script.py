import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def require(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available top-level keys: {list(d.keys())}")
    return d[key]


def get_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    return None


def is_keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    name = str(s.name).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return any(tok in name for tok in ("id", "key", "index", "row", "code"))
    return False


def encode_target(y, positive_label=None):
    y = y.copy()
    if y.isna().any():
        raise ValueError("Target contains missing values")
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(y.unique())
        if len(vals) != 2:
            raise ValueError("Target must be binary")
        if set(vals) == {0, 1}:
            return y.astype(int), 1
        pos = max(vals)
        return (y == pos).astype(int), pos
    norm = y.astype(str).str.strip().str.rstrip(".")
    vals = sorted(norm.unique())
    if len(vals) != 2:
        raise ValueError("Target must be binary")
    if positive_label is not None:
        pos = str(positive_label).strip().rstrip(".")
        if pos not in vals:
            raise ValueError("positive_label not in target values")
    else:
        pos_candidates = [v for v in vals if ">50" in v.lower() or ">50k" in v.lower()]
        pos = pos_candidates[0] if pos_candidates else vals[-1]
    return (norm == pos).astype(int), pos


mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = require(protocol, "targets")
excluded = require(protocol, "excluded_columns")
group = require(protocol, "group_column")
task = require(protocol, "task")
_ = require(task, "primary_metric")
positive_label = task.get("positive_label") or protocol.get("positive_label")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_PDWSHJGW/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

raw_features = [
    c
    for c in train.columns
    if c not in targets
    and c not in excluded
    and c != group
    and not is_keylike(train[c])
]

pre_names = get_names(pre)
if pre_names is not None:
    bad = set(pre_names) & (set(targets) | set(excluded) | ({group} if group is not None else set()))
    if bad:
        raise ValueError(f"Preprocessor expects excluded columns: {sorted(bad)}")
    missing = [c for c in pre_names if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects missing columns: {missing}")
    features = pre_names
else:
    features = raw_features

X = train[features].copy()
y, positive = encode_target(train[targets[0]], positive_label)
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
sel_names = get_names(sel)

if (
    isinstance(sel, Pipeline)
    and sel_names is not None
    and set(sel_names) == set(features)
    and len(sel.named_steps) > 1
):
    steps = list(sel.steps)
    if steps and steps[-1][0] in ("model", "classifier", "clf"):
        steps = steps[:-1]
    pipeline = Pipeline(steps=steps + [("model", clf)])
else:
    if sel_names is not None and set(sel_names) == set(features):
        raise ValueError("Selector expects raw features but does not include preprocessing")
    try:
        Xt = pre.transform(X.head(5))
        if Xt.shape[1] <= 300:
            pre.set_output(transform="pandas")
    except Exception:
        pass
    pipeline = Pipeline([("preprocessor", pre), ("selector", sel), ("model", clf)])

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"Train rows: {len(X)}")
print(f"Features: {len(features)}")
print(f"Positive label: {positive}")
print("Saved model.pkl")
