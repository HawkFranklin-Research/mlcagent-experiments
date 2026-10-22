import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

train = pd.read_parquet(inputs["train"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_K8ATBX97/selector.pkl"])

def feature_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    return list(names) if names is not None else None

raw_cols = []
n = len(train)
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if n > 0 and s.nunique(dropna=True) == n:
            continue
    if pd.api.types.is_integer_dtype(s.dtype) and n > 2 and s.nunique(dropna=True) == n:
        continue
    raw_cols.append(c)

pre_names = feature_names(pre)
sel_names = feature_names(sel)
if pre_names is not None and set(pre_names) == set(raw_cols):
    raw_cols = list(pre_names)

def selector_wraps_preprocessing(sel_obj):
    if not isinstance(sel_obj, Pipeline):
        return False
    keys = [str(k).lower() for k in sel_obj.named_steps.keys()]
    if any(k in ("preprocessor", "preprocess", "preprocessing", "column_transformer", "ct", "transform") for k in keys):
        return True
    first = list(sel_obj.named_steps.values())[0]
    selector_classes = {
        "SelectKBest", "SelectFromModel", "RFE", "RFECV", "VarianceThreshold",
        "SelectPercentile", "GenericUnivariateSelect", "SequentialFeatureSelector"
    }
    return first.__class__.__name__ not in selector_classes

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel is not None and selector_wraps_preprocessing(sel):
    feature_cols = sel_names if sel_names is not None else raw_cols
    pipeline = Pipeline([("selector", sel), ("model", model)])
else:
    feature_cols = pre_names if pre_names is not None else raw_cols
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", model))
    pipeline = Pipeline(steps)

feature_cols = [c for c in feature_cols if c not in drop]
if not feature_cols:
    raise ValueError("No feature columns available")
if any(c in targets or c in excluded for c in feature_cols):
    raise ValueError("Target or excluded column present in feature columns")

X = train.loc[:, feature_cols]
y = train[targets[0]]
fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print("saved model.pkl")
