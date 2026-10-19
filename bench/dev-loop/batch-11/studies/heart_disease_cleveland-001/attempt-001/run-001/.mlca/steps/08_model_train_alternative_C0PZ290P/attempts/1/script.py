import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
metric = protocol["task"]["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["07_model_train_logistic_AGQDNKZV/model.pkl"], "rb") as f:
    prev = pickle.load(f)
assert isinstance(prev, Pipeline), f"previous model is not a Pipeline: {type(prev)}"

if hasattr(prev, "feature_names_in_"):
    drop = set(targets) | set(excluded)
    feat_cols = [c for c in prev.feature_names_in_ if c in train.columns and c not in drop]
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feat_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object and s.nunique() == len(train):
            continue
        feat_cols.append(c)

X_train = train[feat_cols]
X_val = val[feat_cols]
y_train = train[target].values
y_val = val[target].values

vals, counts = np.unique(y_train, return_counts=True)
if len(counts) > 1 and counts.min() / counts.sum() < 0.10:
    clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
else:
    clf = HistGradientBoostingClassifier(random_state=42)

last_name = list(prev.named_steps.keys())[-1]
pipeline = Pipeline(list(prev.steps[:-1]) + [(last_name, clf)])

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

proba = fitted.predict_proba(X_val)
classes = list(fitted.classes_)
pos_label = classes[1] if len(classes) == 2 else classes[-1]
y_true = (y_val == pos_label).astype(int)
auc = float(roc_auc_score(y_true, proba[:, classes.index(pos_label)]))
mlca_runtime.record_metric(metric, auc, "val")

mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"features={len(feat_cols)} classes={classes} val_{metric}={auc:.4f}")
