import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "task", "excluded_columns"]:
    if k not in protocol:
        raise KeyError(f"missing {k}; available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y_train = train[target]
y_val = val[target]

with open(inputs["latest_model"], "rb") as f:
    base = pickle.load(f)

if isinstance(base, Pipeline):
    steps = list(base.steps[:-1])
    feature_cols = list(base.feature_names_in_) if hasattr(base, "feature_names_in_") else \
        [c for c in train.columns if c not in set(targets) | set(excluded) and c != group]
else:
    steps = []
    feature_cols = [c for c in train.columns if c not in set(targets) | set(excluded) and c != group]

counts = y_train.value_counts(normalize=True)
cw = "balanced" if counts.min() < 0.1 else None
hgb = HistGradientBoostingClassifier(random_state=42, class_weight=cw)
pipe = Pipeline(steps + [("model", hgb)])

X_train = train[feature_cols]
X_val = val[feature_cols]
fresh = mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(fresh, "alternative_model.pkl")

if primary == "roc_auc" and len(np.unique(y_val)) == 2:
    model = fresh.named_steps["model"]
    classes = model.classes_
    proba = fresh.predict_proba(X_val)
    y_bin = (y_val == classes[1]).astype(int)
    auc = roc_auc_score(y_bin, proba[:, 1])
    mlca_runtime.record_metric("roc_auc", float(auc), "val")
    print(f"val roc_auc={auc:.4f}")
print(f"train={train.shape} features={len(feature_cols)} class_weight={cw}")
