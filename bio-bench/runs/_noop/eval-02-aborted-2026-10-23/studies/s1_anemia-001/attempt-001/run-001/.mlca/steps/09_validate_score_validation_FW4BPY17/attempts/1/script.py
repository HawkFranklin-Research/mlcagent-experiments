import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol.get("targets") or [])
excluded = set(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
primary_metric = task["primary_metric"]
task_type = task["type"]
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

keylike = set()
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if s.dtype == object and s.nunique() == len(s):
        keylike.add(c)
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded
                and c != group_col and c not in keylike]

model_paths = {
    "logistic": inputs["07_model_train_logistic_Y2147J2H/model.pkl"],
    "alternative": inputs["08_model_train_alternative_3WXQ24M0/alternative_model.pkl"],
}
models = {}
for name, p in model_paths.items():
    with open(p, "rb") as f:
        models[name] = pickle.load(f)

def get_input_cols(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return feature_cols

def get_classes(m):
    if hasattr(m, "classes_"):
        return m.classes_
    if hasattr(m, "named_steps"):
        return list(m.named_steps.values())[-1].classes_
    raise RuntimeError("model has no classes_")

y_val = val[target_col].values
rows = []
best = {"name": None, "score": -np.inf}
for name, m in models.items():
    cols = get_input_cols(m)
    X = val[cols]
    proba = m.predict_proba(X)
    classes = get_classes(m)
    pos_idx = 1
    pos_label = classes[pos_idx]
    p_pos = proba[:, pos_idx]
    y_bin = (y_val == pos_label).astype(int)
    roc = float(roc_auc_score(y_bin, p_pos))
    pr = float(average_precision_score(y_bin, p_pos))
    fpr, tpr, thr = roc_curve(y_bin, p_pos)
    j = tpr - fpr
    best_thr = float(thr[int(np.argmax(j))])
    y_pred = (p_pos >= best_thr).astype(int)
    bal = float(balanced_accuracy_score(y_bin, y_pred))
    brier = float(brier_score_loss(y_bin, p_pos))
    for metric, value in [("roc_auc", roc), ("pr_auc", pr),
                          ("balanced_accuracy", bal), ("brier_score", brier)]:
        rows.append({"model": name, "metric": metric,
                     "value": round(value, 6),
                     "threshold": round(best_thr, 6)})
    if roc > best["score"]:
        best = {"name": name, "score": roc}

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation metrics per candidate; Youden J threshold on val.")

mlca_runtime.record_metric(primary_metric, best["score"], "val")
mlca_runtime.record_fact_candidate("best_model", best["name"],
                                    source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best["score"],
                                    source="validation_metrics.csv")

print(f"val rows={len(val)} candidates={len(models)}")
print(f"best_model={best['name']} {primary_metric}={best['score']:.4f}")
