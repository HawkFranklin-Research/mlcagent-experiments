import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col]
feat_cols = [c for c in feat_cols if not is_keylike(train[c])]

y_val = val[target].to_numpy()

model_keys = [
    ("logistic", "07_model_train_logistic_D8PWVNVS/model.pkl"),
    ("alternative", "08_model_train_alternative_V3H8V8QN/alternative_model.pkl"),
    ("latest", "latest_model"),
]
models = {}
for name, key in model_keys:
    with open(inputs[key], "rb") as f:
        models[name] = pickle.load(f)

def input_cols(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return feat_cols

rows = []
best_name, best_auc = None, -1.0
for name, model in models.items():
    cols = input_cols(model)
    X_val = val[cols]
    proba = model.predict_proba(X_val)
    classes = list(model.classes_)
    pos = classes.index(1) if 1 in classes else 1
    p = proba[:, pos]
    auc = float(roc_auc_score(y_val, p))
    ap = float(average_precision_score(y_val, p))
    brier = float(brier_score_loss(y_val, p))
    fpr, tpr, thr = roc_curve(y_val, p)
    idx = int(np.argmax(tpr - fpr))
    th = float(thr[idx])
    pred = (p >= th).astype(int)
    ba = float(balanced_accuracy_score(y_val, pred))
    tp = int(((pred == 1) & (y_val == 1)).sum()); fn = int(((pred == 0) & (y_val == 1)).sum())
    tn = int(((pred == 0) & (y_val == 0)).sum()); fp = int(((pred == 1) & (y_val == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    for m, v, t in [("roc_auc", auc, np.nan), ("pr_auc", ap, np.nan),
                    ("balanced_accuracy", ba, th), ("brier_score", brier, np.nan),
                    ("sensitivity", sens, th), ("specificity", spec, th)]:
        rows.append({"model": name, "metric": m,
                     "value": round(float(v), 4), "threshold": t})
    mlca_runtime.record_metric(f"roc_auc_{name}", auc, "val")
    if auc > best_auc:
        best_auc, best_name = auc, name

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(f"models={len(models)} val_rows={len(val)} best={best_name} auc={best_auc:.4f}")
