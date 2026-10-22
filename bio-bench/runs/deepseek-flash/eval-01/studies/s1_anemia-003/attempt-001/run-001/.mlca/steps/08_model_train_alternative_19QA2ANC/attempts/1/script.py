import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime as rt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rt.set_seed(42)
inputs = rt.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
task = proto["task"]
y_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["07_model_train_logistic_Q726PM5C/model.pkl"], "rb") as f:
    base = pickle.load(f)

if not isinstance(base, Pipeline):
    raise TypeError(f"Expected a Pipeline at 07 model, got {type(base).__name__}")

feat_in = list(base.feature_names_in_) if hasattr(base, "feature_names_in_") else None
if feat_in is None:
    feat_in = [c for c in train.columns if c not in set(targets)]
Xtr = train[feat_in]
ytr = train[y_col]

counts = ytr.value_counts()
imb = counts.min() / counts.sum()
kw = {"random_state": 42}
if imb < 0.1:
    kw["class_weight"] = "balanced"

clf = HistGradientBoostingClassifier(**kw)
steps = list(base.steps[:-1]) + [("clf", clf)]
pipe = Pipeline(steps)

rt.fit_record(pipe, Xtr, "train", y=ytr)

classes = pipe.named_steps["clf"].classes_
pos = classes[1]
proba = pipe.predict_proba(Xtr)[:, 1]
auc = roc_auc_score((ytr == pos).astype(int), proba)
rt.record_metric("train_roc_auc", float(auc), "train")

rt.save_model(pipe, "alternative_model.pkl")

print("steps:", list(pipe.named_steps.keys()))
print("n_features:", len(feat_in))
print("class_imbalance:", round(float(imb), 4), "class_weight_set:", "class_weight" in kw)
print("train_roc_auc:", round(float(auc), 4))
