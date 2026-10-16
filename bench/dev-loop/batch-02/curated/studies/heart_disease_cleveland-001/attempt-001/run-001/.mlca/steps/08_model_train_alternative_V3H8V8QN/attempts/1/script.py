import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["latest_model"], "rb") as f:
    saved = pickle.load(f)
if not isinstance(saved, Pipeline):
    raise TypeError("latest_model is not a Pipeline")

if hasattr(saved, "feature_names_in_"):
    feat_cols = list(saved.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feat_cols = [c for c in train.columns if c not in drop]

X_train = train[feat_cols]
y_train = train[targets[0]]

steps = list(saved.steps)
steps[-1] = ("clf", HistGradientBoostingClassifier(random_state=42))
pipe = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

proba = fitted.predict_proba(val[feat_cols])
classes = list(fitted.classes_)
pos = classes.index(1) if 1 in classes else 1
auc = roc_auc_score(val[targets[0]], proba[:, pos])
mlca_runtime.record_metric("roc_auc", float(auc), "val")
print(f"train={X_train.shape} val={val.shape} roc_auc={auc:.4f}")
