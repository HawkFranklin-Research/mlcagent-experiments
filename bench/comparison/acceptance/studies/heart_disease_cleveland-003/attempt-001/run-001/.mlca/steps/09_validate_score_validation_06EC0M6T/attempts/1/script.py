import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")

task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary_metric = task["primary_metric"]
task_type = task.get("type", "binary")
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y_val = val[target]

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group_col is not None and c == group_col):
        continue
    s = train[c]
    all_distinct = s.nunique(dropna=False) == len(train)
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    is_int = pd.api.types.is_integer_dtype(s)
    if all_distinct and (is_text or is_int) and not pd.api.types.is_float_dtype(s):
        continue
    feature_cols.append(c)

def expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline):
        first = model.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

candidate_paths = [
    ("logistic", inputs["07_model_train_logistic_H9QQPD1P/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_MAM5EC9Z/alternative_model.pkl"]),
]
latest = inputs.get("latest_model")
if latest is not None and all(latest.resolve() != p.resolve() for _, p in candidate_paths):
    candidate_paths.append(("latest", latest))

candidate_metrics = {}
candidate_thresholds = {}
rows = []

for name, path in candidate_paths:
    model = load_model(path)
    exp = expected_columns(model)
    if exp is None:
        X = val[feature_cols]
    else:
        missing = [c for c in exp if c not in feature_cols]
        if missing:
            raise ValueError(f"Model {name} expects non-feature columns: {missing}")
        X = val[exp]

    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = model.steps[-1][1].classes_
    classes = np.asarray(classes)

    if n_classes == 2:
        pos_label = task.get("positive_label", classes[1])
        pos_idx = list(classes).index(pos_label)
        prob_pos = proba[:, pos_idx]
        y_bin = (y_val == pos_label).astype(int)

        roc = roc_auc_score(y_bin, prob_pos)
        pr = average_precision_score(y_bin, prob_pos)
        fpr, tpr, thr = roc_curve(y_bin, prob_pos)
        finite = np.isfinite(thr)
        if finite.any():
            j = tpr[finite] - fpr[finite]
            threshold = float(thr[finite][int(np.argmax(j))])
        else:
            threshold = 0.5
        pred = (prob_pos >= threshold).astype(int)
        bal = balanced_accuracy_score(y_bin, pred)
        brier = brier_score_loss(y_bin, prob_pos)

        metrics = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier_score": brier}
        thresholds = {"roc_auc": np.nan, "pr_auc": np.nan, "balanced_accuracy": threshold, "brier_score": np.nan}
    else:
        roc_ovr = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
        pred = model.predict(X)
        f1_macro = f1_score(y_val, pred, average="macro", labels=classes, zero_division=0)
        metrics = {"roc_auc_ovr": roc_ovr, "f1_macro": f1_macro}
        thresholds = {"roc_auc_ovr": np.nan, "f1_macro": np.nan}

    if primary_metric not in metrics:
        raise ValueError(f"Primary metric {primary_metric} not computed; available: {list(metrics)}")

    candidate_metrics[name] = metrics
    candidate_thresholds[name] = thresholds
    for metric, value in metrics.items():
        rows.append({
            "model": name,
            "metric": metric,
            "value": float(value),
            "threshold": thresholds[metric],
        })

higher_better = {
    "roc_auc": True, "pr_auc": True, "balanced_accuracy": True,
    "roc_auc_ovr": True, "f1_macro": True, "brier_score": False,
}
higher = higher_better.get(primary_metric, True)
if higher:
    best_name = max(candidate_metrics, key=lambda n: candidate_metrics[n][primary_metric])
else:
    best_name = min(candidate_metrics, key=lambda n: candidate_metrics[n][primary_metric])
best_auc = candidate_metrics[best_name].get("roc_auc", candidate_metrics[best_name].get("roc_auc_ovr"))

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

out = pd.DataFrame(rows)
out["value"] = out["value"].map(sig4)
out["threshold"] = out["threshold"].map(sig4)
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics by candidate model")

order = sorted(candidate_metrics, key=lambda n: candidate_metrics[n][primary_metric], reverse=not higher)
for name in order:
    mlca_runtime.record_metric(primary_metric, float(candidate_metrics[name][primary_metric]), "val")

mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best_model={best_name} val_{primary_metric}={candidate_metrics[best_name][primary_metric]:.4f}")
print(f"saved validation_metrics.csv rows={len(out)}")
