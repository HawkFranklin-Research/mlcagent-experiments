import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["06_preprocess_fit_preprocessing_MYPQXFW6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_NK102NGP/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(est):
    if isinstance(est, Pipeline) and est.steps:
        return input_names(est.steps[0][1])
    names = getattr(est, "feature_names_in_", None)
    return list(names) if names is not None else None

base_excl = set(targets) | excluded | ({group_col} if group_col else set())
allowed = [c for c in train.columns if c not in base_excl]
feature_cols = []
for c in allowed:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s) and len(s) > 1:
        continue
    if any(k in c.lower() for k in ("id", "key", "index")) and s.nunique(dropna=False) == len(s) and len(s) > 1:
        continue
    feature_cols.append(c)

# Check whether the selector already contains a raw-data frontend.
sel_is_frontend = False
if isinstance(sel, Pipeline) and sel.steps:
    names = input_names(sel.steps[0][1])
    if names and (set(names).issubset(set(feature_cols)) or set(feature_cols).issubset(set(names))):
        sel_is_frontend = True
        feature_cols = [c for c in names if c in set(feature_cols)]

pre_has_selector = isinstance(pre, Pipeline) and any(
    k in pre.named_steps for k in ("selector", "select", "feature_selection")
)

if not sel_is_frontend:
    names = input_names(pre)
    if names:
        feature_cols = [c for c in names if c in set(feature_cols)]

y_train = train[targets[0]]
clf = HistGradientBoostingClassifier(random_state=seed)
if y_train.value_counts(normalize=True).min() < 0.1:
    clf.set_params(class_weight="balanced")

if sel_is_frontend:
    steps = list(sel.steps) + [("classifier", clf)]
elif pre_has_selector:
    steps = [("preprocessor", pre), ("classifier", clf)]
else:
    steps = [("preprocessor", pre), ("selector", sel), ("classifier", clf)]

pipe = Pipeline(steps)
X_train = train[feature_cols]
fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

try:
    X_val = val[feature_cols]
    y_val = val[targets[0]]
    proba = fitted.predict_proba(X_val)
    classes = fitted.named_steps["classifier"].classes_
    score = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    mlca_runtime.record_metric("roc_auc_ovr", float(score), "val")
    print("val roc_auc_ovr", round(float(score), 4))
except Exception as e:
    print("metric skipped:", type(e).__name__)

print("features", len(feature_cols), "train", X_train.shape)
