import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["07_model_train_logistic_SGR96899/model.pkl"], "rb") as f:
    base = pickle.load(f)

if not isinstance(base, Pipeline):
    raise TypeError("Expected the saved logistic model to be a sklearn Pipeline")

min_frac = train[target].value_counts(normalize=True).min()
hgb_kwargs = {"random_state": seed}
if min_frac < 0.10:
    hgb_kwargs["class_weight"] = "balanced"
hgb = HistGradientBoostingClassifier(**hgb_kwargs)

steps = base.steps
last_name = steps[-1][0]
pipe = Pipeline(steps[:-1] + [(last_name, hgb)])

cols = list(getattr(base, "feature_names_in_", []))
if not cols:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    cols = [c for c in train.columns if c not in drop]

X_train = train[cols].copy()
y_train = train[target]
mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

if primary_metric == "roc_auc" and len(pipe.classes_) == 2:
    X_val = val[cols].copy()
    proba = pipe.predict_proba(X_val)
    pos_idx = 1
    y_bin = (val[target] == pipe.classes_[pos_idx]).astype(int)
    auc = roc_auc_score(y_bin, proba[:, pos_idx])
    mlca_runtime.record_metric("roc_auc", float(auc), "val")
    print("val_roc_auc", round(float(auc), 4))

print("train", X_train.shape)
print("features", len(cols))
print("classes", pipe.classes_.tolist())
print("saved alternative_model.pkl")
