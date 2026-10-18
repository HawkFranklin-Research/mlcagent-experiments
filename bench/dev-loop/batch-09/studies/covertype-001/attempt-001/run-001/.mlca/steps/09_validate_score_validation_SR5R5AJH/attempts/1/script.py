import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
metric_name = task["primary_metric"]
if metric_name != "roc_auc_ovr":
    raise ValueError(f"Expected roc_auc_ovr, got {metric_name}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]
X_val = val[feature_cols]
target_col = targets[0]
y_val = val[target_col]

models = {
    "logistic": pickle.load(open(inputs["07_model_train_logistic_JDJV5YWV/model.pkl"], "rb")),
    "alternative": pickle.load(open(inputs["08_model_train_alternative_DE6J0FH7/alternative_model.pkl"], "rb")),
}

rows = []
aucs = {}
for name, model in models.items():
    proba = model.predict_proba(X_val)
    classes = np.asarray(model.classes_)
    auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    pred = classes[proba.argmax(axis=1)]
    f1 = f1_score(y_val, pred, average="macro")
    aucs[name] = float(auc)
    rows.append({"model": name, "metric": metric_name, "value": float(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f1), "threshold": np.nan})
    mlca_runtime.record_metric(metric_name, float(auc), "val")

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per candidate model.")

best = max(aucs, key=aucs.get)
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", aucs[best], source="validation_metrics.csv")

print(f"n_val={len(val)} n_features={len(feature_cols)}")
print(f"aucs={aucs} best={best}")
