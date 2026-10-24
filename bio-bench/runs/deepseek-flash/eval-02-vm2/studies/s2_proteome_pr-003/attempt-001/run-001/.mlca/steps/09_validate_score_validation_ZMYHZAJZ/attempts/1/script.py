import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve,
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

if "task" not in protocol:
    raise ValueError(f"_protocol missing 'task'; available top-level keys: {list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise ValueError(f"_protocol['task'] missing 'primary_metric'; available top-level keys: {list(protocol.keys())}")
if "targets" not in protocol:
    raise ValueError(f"_protocol missing 'targets'; available top-level keys: {list(protocol.keys())}")

primary = task["primary_metric"]
targets = list(protocol["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}; available top-level keys: {list(protocol.keys())}")
label = targets[0]

val = pd.read_parquet(inp["val"])
y = val[label]

with open(inp["05_preprocess_fit_preprocessing_YCEBPQHT/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_FXHKDZPP/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inp["07_model_train_logistic_VFJ1YWNX/model.pkl"], "rb") as f:
    model_log = pickle.load(f)
with open(inp["08_model_train_alternative_7GE7PGA8/alternative_model.pkl"], "rb") as f:
    model_alt = pickle.load(f)

models = {"logistic": model_log, "alternative": model_alt}


def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = next(iter(obj.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None


def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if isinstance(model, Pipeline):
        return np.asarray(next(reversed(list(model.named_steps.values()))).classes_)
    raise ValueError("Model has no classes_ attribute")


def get_proba(model, df):
    cols = input_cols(model)
    if cols is not None:
        return model.predict_proba(df[cols])
    if selector is not None and hasattr(selector, "feature_names_in_"):
        cols = list(selector.feature_names_in_)
        return model.predict_proba(selector.transform(df[cols]))
    if selector is not None:
        cols = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else [c for c in df.columns if c != label]
        xp = pre.transform(df[cols]) if pre is not None else df[cols]
        return model.predict_proba(selector.transform(xp))
    cols = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else [c for c in df.columns if c != label]
    xp = pre.transform(df[cols]) if pre is not None else df[cols]
    return model.predict_proba(xp)


def sig4(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")


direction = {
    "roc_auc": "max", "pr_auc": "max", "balanced_accuracy": "max",
    "brier_score": "min",
}.get(primary)
if direction is None:
    raise ValueError(f"Unsupported primary_metric {primary!r}; available top-level keys: {list(protocol.keys())}")

rows = []
primary_by_model = {}
roc_by_model = {}
positive_labels = {}

for name, model in models.items():
    classes = get_classes(model)
    pos = task.get("positive_label", None)
    if pos is None:
        pos = classes[1]
    if pos not in classes:
        raise ValueError(f"Positive label {pos!r} not in model classes {classes!r}")
    pos_idx = int(np.where(classes == pos)[0][0])

    proba = np.asarray(get_proba(model, val))
    p_pos = proba if proba.ndim == 1 else proba[:, pos_idx]
    y_bin = (y == pos).astype(int)
    if y_bin.nunique() < 2:
        raise ValueError("Validation target has fewer than two classes")

    auc = roc_auc_score(y_bin, p_pos)
    ap = average_precision_score(y_bin, p_pos)
    fpr, tpr, thr = roc_curve(y_bin, p_pos)
    valid = np.isfinite(thr)
    if valid.any():
        j = int(np.argmax((tpr - fpr)[valid]))
        threshold = float(thr[valid][j])
    else:
        threshold = np.nan
    y_hat = (p_pos >= threshold).astype(int) if np.isfinite(threshold) else np.zeros(len(y_bin), dtype=int)
    ba = balanced_accuracy_score(y_bin, y_hat)
    brier = brier_score_loss(y_bin, p_pos)

    metric_vals = {
        "roc_auc": auc, "pr_auc": ap,
        "balanced_accuracy": ba, "brier_score": brier,
    }
    primary_by_model[name] = metric_vals[primary]
    roc_by_model[name] = auc
    positive_labels[name] = str(pos)

    for metric, value in metric_vals.items():
        metric_name = primary if metric == primary else metric
        row_threshold = threshold if metric == "balanced_accuracy" else np.nan
        rows.append({
            "model": name,
            "metric": metric_name,
            "value": sig4(value),
            "threshold": sig4(row_threshold),
        })

if direction == "max":
    best_name = max(primary_by_model, key=primary_by_model.get)
else:
    best_name = min(primary_by_model, key=primary_by_model.get)
best_primary = primary_by_model[best_name]
best_roc_auc = roc_by_model[best_name]

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics for candidate models.")

mlca_runtime.record_metric(primary, float(best_primary), split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_roc_auc), source="validation_metrics.csv")

print(f"validation rows: {len(val)}")
print(f"best_model: {best_name}")
print(f"{primary}: {best_primary:.4g}")
