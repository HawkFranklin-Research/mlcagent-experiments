import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, found {targets}")
target = targets[0]
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
_ = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"Target {target!r} not in training table")
y = train[target]

model_path = inputs["latest_model"]
with open(model_path, "rb") as f:
    base = pickle.load(f)

def names_of(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        first = obj.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

feature_names = names_of(base)
drop = set(targets + excluded + ([group] if group is not None else []))
if feature_names is None:
    feature_names = []
    n = len(train)
    for col in train.columns:
        if col in drop:
            continue
        s = train[col]
        low = col.lower()
        if low in {"id", "row_id", "index", "patient_id"} or low.endswith("_id"):
            continue
        if s.nunique(dropna=True) == n and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            continue
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ) and s.nunique(dropna=True) > max(5, int(0.05 * n)):
            continue
        feature_names.append(col)

feature_names = [c for c in feature_names if c in train.columns and c not in drop]
if not feature_names:
    raise ValueError("No usable feature columns")
if set(feature_names) & drop:
    raise ValueError("Target or excluded column reached the feature set")

minority = y.value_counts(normalize=True).min()
hgb_kwargs = {"random_state": seed}
if minority < 0.10:
    hgb_kwargs["class_weight"] = "balanced"
hgb = HistGradientBoostingClassifier(**hgb_kwargs)

if isinstance(base, Pipeline):
    steps = list(base.steps)
    last_name = steps[-1][0]
    alternative = Pipeline(steps[:-1] + [(last_name, hgb)])
else:
    pre_path = inputs["05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl"]
    sel_path = inputs["06_features_select_features_BD9SGMPY/selector.pkl"]
    with open(pre_path, "rb") as f:
        pre = pickle.load(f)
    with open(sel_path, "rb") as f:
        sel = pickle.load(f)
    sel_names = names_of(sel)
    if sel_names is not None and set(sel_names) == set(feature_names):
        alternative = Pipeline([("selector", sel), ("classifier", hgb)])
    else:
        alternative = Pipeline([("preprocessor", pre), ("selector", sel), ("classifier", hgb)])

X = train[feature_names]
fitted = mlca_runtime.fit_record(alternative, X, "train", y=y)
if fitted is None:
    fitted = alternative
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"trained={type(fitted).__name__} rows={len(X)} features={X.shape[1]}")
