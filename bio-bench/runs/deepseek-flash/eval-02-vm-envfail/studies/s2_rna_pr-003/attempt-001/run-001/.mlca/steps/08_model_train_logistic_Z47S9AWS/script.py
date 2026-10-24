import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise ValueError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded_raw = protocol["excluded_columns"]
if not isinstance(excluded_raw, (list, tuple, set)):
    raise ValueError(f"excluded_columns is not a list; top-level keys: {list(protocol.keys())}")
excluded = set(map(str, excluded_raw))
group_col = protocol.get("group_column")
task = protocol["task"]
metric_name = task.get("primary_metric", "roc_auc")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

base_features = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
]
key_like = []
for c in base_features:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=False) == len(s):
        key_like.append(c)
base_features = [c for c in base_features if c not in key_like]

with open(inputs["06_preprocess_fit_preprocessing_1X70GNWA/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_41RKQQBB/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(c) for c in step.feature_names_in_]
    return None

def output_names(obj):
    try:
        return [str(c) for c in obj.get_feature_names_out()]
    except Exception:
        return None

pre_in = input_names(pre)
feature_cols = pre_in if pre_in is not None else base_features
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Feature columns missing from train: {missing[:5]}")
bad = set(feature_cols) & (set(targets) | excluded | ({group_col} if group_col else set()))
if bad:
    raise ValueError(f"Target/excluded/group columns in features: {sorted(bad)[:5]}")

sel_in = input_names(sel)
pre_out = output_names(pre)
sel_is_pipeline = isinstance(sel, Pipeline) and len(sel.named_steps) > 1
logit = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel_is_pipeline and (sel_in is None or sel_in == feature_cols):
    pipe = Pipeline([("frontend", sel), ("model", logit)])
elif sel_in is not None and pre_out is not None and sel_in == pre_out:
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", logit)])
elif sel_in == feature_cols and pre_out == feature_cols:
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", logit)])
elif sel_in == feature_cols:
    pipe = Pipeline([
        ("features", FeatureUnion([("pre", pre), ("sel", sel)])),
        ("model", logit),
    ])
else:
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", logit)])

X_train = train[feature_cols]
y_train = train[targets[0]]
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(pipe, "model.pkl")

auc = None
try:
    classes = list(pipe.named_steps["model"].classes_)
    positive = task.get("positive_label", classes[1] if len(classes) > 1 else classes[0])
    j = classes.index(positive)
    proba = pipe.predict_proba(val[feature_cols])[:, j]
    y_bin = (val[targets[0]] == positive).astype(int)
    auc = float(roc_auc_score(y_bin, proba))
    mlca_runtime.record_metric(metric_name, auc, "val")
except Exception as exc:
    print(f"validation metric skipped: {type(exc).__name__}")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
if auc is not None:
    print(f"val {metric_name}={auc:.4f}")
