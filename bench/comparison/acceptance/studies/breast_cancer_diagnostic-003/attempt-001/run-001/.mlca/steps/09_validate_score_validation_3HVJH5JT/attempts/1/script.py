import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_curve,
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available: {list(d)}")
    return d[k]

task = req(protocol, "task")
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = protocol.get("group_column")
primary = str(req(task, "primary_metric"))
task_type = str(task.get("type", "")).lower()

protected = set(targets) | set(excluded) | ({group} if group else set())

def keylike(s):
    n = s.nunique(dropna=False)
    if s.dtype.kind in "OUS" and n == len(s):
        return True
    if s.dtype.kind in "iu" and n == len(s) and any(t in str(s.name).lower() for t in ("id", "code", "key", "index")):
        return True
    return False

feature_cols = [c for c in train.columns if c not in protected and not keylike(train[c])]
if len(targets) != 1:
    raise ValueError("This validation script expects exactly one target")
ycol = targets[0]
y_val = val[ycol]

candidates = [
    ("logistic_QMMJ5EQH", inputs["07_model_train_logistic_QMMJ5EQH/model.pkl"]),
    ("alternative_3P2DK38C", inputs["08_model_train_alternative_3P2DK38C/alternative_model.pkl"]),
]

def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if isinstance(model, Pipeline):
        last = list(model.named_steps.values())[-1]
        return np.asarray(last.classes_)
    raise AttributeError("Model has no classes_")

def binary_metrics(model, X, y):
    classes = get_classes(model)
    pos = task.get("positive_label", classes[1])
    pos_idx = int(np.where(classes == pos)[0][0])
    prob = np.asarray(model.predict_proba(X))[:, pos_idx]
    y_bin = (np.asarray(y) == pos).astype(int)
    roc = roc_auc_score(y_bin, prob)
    pr = average_precision_score(y_bin, prob)
    fpr, tpr, thr = roc_curve(y_bin, prob)
    j = tpr - fpr
    finite = np.isfinite(thr)
    idx = int(np.argmax(np.where(finite, j, -np.inf)))
    threshold = float(thr[idx])
    pred_bin = (prob >= threshold).astype(int)
    neg = classes[0] if pos == classes[1] else classes[1]
    pred_labels = np.where(pred_bin == 1, pos, neg)
    bal = balanced_accuracy_score(np.asarray(y), pred_labels)
    brier = brier_score_loss(y_bin, prob)
    return {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier": brier}, threshold, roc

def multiclass_metrics(model, X, y):
    classes = get_classes(model)
    proba = np.asarray(model.predict_proba(X))
    pred = np.asarray(model.predict(X))
    roc = roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro")
    f1 = f1_score(y, pred, average="macro")
    return {"roc_auc_ovr": roc, "roc_auc": roc, "macro_f1": f1, "f1_macro": f1}, roc

rows, results = [], []
for name, path in candidates:
    model = load_obj(path)
    cols = list(getattr(model, "feature_names_in_", feature_cols))
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"Missing validation columns for {name}: {missing}")
    X = val.loc[:, cols]
    if task_type == "binary":
        mv, threshold, roc = binary_metrics(model, X, y_val)
        alias = {"roc_auc_score": "roc_auc", "pr_auc_score": "pr_auc", "brier_score": "brier"}
        pkey = alias.get(primary.lower().replace(" ", "_"), primary.lower().replace(" ", "_"))
        if pkey not in mv:
            raise KeyError(f"Unsupported primary metric {primary}; available: {list(mv)}")
        pv = mv[pkey]
        for k in ["roc_auc", "pr_auc", "balanced_accuracy", "brier"]:
            rows.append({
                "model": name,
                "metric": primary if k == pkey else k,
                "value": mv[k],
                "threshold": threshold if k == "balanced_accuracy" else np.nan,
            })
        results.append((name, pv, roc))
    else:
        mv, roc = multiclass_metrics(model, X, y_val)
        alias = {"roc_auc": "roc_auc_ovr", "roc_auc_ovr": "roc_auc_ovr", "f1_macro": "macro_f1"}
        pkey = alias.get(primary.lower().replace(" ", "_"), primary.lower().replace(" ", "_"))
        if pkey not in mv:
            raise KeyError(f"Unsupported primary metric {primary}; available: {list(mv)}")
        pv = mv[pkey]
        for k in ["roc_auc_ovr", "macro_f1"]:
            rows.append({
                "model": name,
                "metric": primary if k == pkey else k,
                "value": mv[k],
                "threshold": np.nan,
            })
        results.append((name, pv, roc))
    mlca_runtime.record_metric(primary, float(pv), "val")

lower_is_better = pkey in {"brier", "log_loss", "mae", "mse", "rmse"}
best = min(results, key=lambda t: t[1]) if lower_is_better else max(results, key=lambda t: t[1])
best_name, best_primary, best_auc = best

def round_sig(x):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return np.nan
    return float(f"{float(x):.4g}")

metrics = pd.DataFrame(rows)
metrics["value"] = metrics["value"].map(round_sig)
metrics["threshold"] = metrics["threshold"].map(round_sig)
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics")

mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", round_sig(best_auc), source="validation_metrics.csv")

print(f"task={task_type} primary={primary} val_rows={len(val)} features={len(feature_cols)}")
print(f"candidates={len(results)}")
print(f"best_model={best_name} primary_value={round_sig(best_primary)}")
print(f"best_val_roc_auc={round_sig(best_auc)}")
