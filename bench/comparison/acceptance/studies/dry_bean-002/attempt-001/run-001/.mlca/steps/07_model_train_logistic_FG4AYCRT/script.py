import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
IN = mlca_runtime.inputs()


def require(d, key):
    if key not in d:
        raise KeyError(f"Missing required protocol key {key!r}; available top-level keys: {sorted(d)}")
    return d[key]


protocol = json.loads(IN["_protocol"].read_text())
targets = list(require(protocol, "targets"))
excluded = list(require(protocol, "excluded_columns"))
group = require(protocol, "group_column")
task = require(protocol, "task")
if not isinstance(task, dict) or "primary_metric" not in task:
    raise KeyError(f"task must contain 'primary_metric'; available task keys: {sorted(task) if isinstance(task, dict) else type(task)}")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target_col = targets[0]

train = pd.read_parquet(IN["train"])
if target_col not in train.columns:
    raise ValueError(f"Target {target_col!r} is not in the training table")

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

n_rows = len(train)
feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if n_rows > 1 and s.nunique(dropna=False) == n_rows:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

with open(IN["05_preprocess_fit_preprocessing_NBMSGCPN/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(IN["06_features_select_features_H3ASCQ8N/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def contains_preprocessing(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if "preprocess" in name.lower() or contains_preprocessing(step):
                return True
    return False


def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            names = input_names(step)
            if names is not None:
                return names
    return None


if contains_preprocessing(sel):
    base = sel if isinstance(sel, Pipeline) else Pipeline([("preprocessor", sel)])
    steps = list(base.steps)
elif isinstance(sel, Pipeline):
    steps = [("preprocessor", pre)] + list(sel.steps)
else:
    steps = [("preprocessor", pre), ("selector", sel)]

steps = [(name, step) for name, step in steps if name != "classifier"]
frontend = steps[0][1]
names = input_names(frontend)
forbidden = set(targets) | set(excluded) | ({group} if group is not None else set())

if names is not None:
    missing = [c for c in names if c not in train.columns]
    if missing:
        raise ValueError(f"Saved frontend expects columns absent from train: {missing}")
    bad = [c for c in names if c in forbidden]
    if bad:
        raise ValueError(f"Saved frontend expects forbidden columns: {bad}")
    if any(c not in feature_cols for c in names):
        raise ValueError("Saved frontend expects columns outside the declared feature set")
    input_cols = names
else:
    input_cols = feature_cols

pipeline = Pipeline(steps + [("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
X = train.loc[:, input_cols].copy()
y = train[target_col]
if y.isna().any():
    raise ValueError("Training target contains missing values")

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
if fitted is None:
    fitted = pipeline
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows={len(X)}, features={X.shape[1]}, classes={y.nunique()}")
print("saved model.pkl")
