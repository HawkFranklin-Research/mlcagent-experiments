import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_curve
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        n = str(s.name).lower()
        return any(t in n for t in ("id", "index", "row", "key", "code"))
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]
if not feature_cols:
    raise ValueError("No feature columns after exclusions")

model_paths = {
    "logistic": inputs["07_model_train_logistic_N8YSRX76/model.pkl"],
    "alternative": inputs["08_model_train_alternative_M1N0MCXA/alternative_model.pkl"],
}
models = {name: pickle.load(open(path, "rb")) for name, path in model_paths.items()}

def cols_for(model):
    c = getattr(model, "feature_names_in_", None)
    if c is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                c = step.feature_names_in_
                break
    c = list(c) if c is not None else list(feature_cols)
    if any(x in drop for x in c):
        raise ValueError("Model input includes a target or excluded column")
    if any(x not in feature_cols for x in c):
        raise ValueError("Model expects a feature outside the declared feature columns")
    return c

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

def binary_metrics(model, X, y):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pos = task.get("positive_label", task.get("positive_class"))
    if pos is None:
        pos = classes[1]
    if pos not in classes:
        raise ValueError(f"Positive label {pos!r} not in model classes {classes}")
    p = proba[:, classes.index(pos)]
    yb = (y == pos).astype(int).to_numpy()
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr
    threshold = float(thr[int(np.argmax(j))])
    pred = (p >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(yb, p),
        "pr_auc": average_precision_score(yb, p),
        "balanced_accuracy": balanced_accuracy_score(yb, pred),
        "brier_score": brier_score_loss(yb, p),
    }
    return metrics, threshold, classes

def multiclass_metrics(model, X, y):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pred = model.predict(X)
    present = [c for c in classes if np.any(y.to_numpy() == c)]
    if len(present) == len(classes):
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        vals = [roc_auc_score((y.to_numpy() == c).astype(int), proba[:, j])
                for j, c in enumerate(classes) if c in present]
        auc = float(np.mean(vals))
        print("classes left out", [c for c in classes if c not in present])
    metrics = {
        "roc_auc_ovr": auc,
        "f1_macro": f1_score(y, pred, average="macro", labels=classes),
    }
    return metrics, np.nan, classes

y = val[target]
is_binary = task.get("type") == "binary" or len(np.unique(y)) == 2
rows, records = [], {}
for name, model in models.items():
    X = val[cols_for(model)]
    if is_binary:
        metrics, threshold, _ = binary_metrics(model, X, y)
        metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]
    else:
        metrics, threshold, _ = multiclass_metrics(model, X, y)
        metric_names = ["roc_auc_ovr", "f1_macro"]
    if primary not in metrics:
        raise ValueError(f"Primary metric {primary!r} not computed; available: {list(metrics)}")
    records[name] = (metrics, threshold)
    for metric in metric_names:
        rows.append({
            "model": name,
            "metric": metric,
            "value": sig4(metrics[metric]),
            "threshold": threshold if is_binary else np.nan,
        })
    mlca_runtime.record_metric(primary, float(metrics[primary]), split="val")

table = pd.DataFrame(rows)
mlca_runtime.save_table(table, "validation_metrics.csv", caption="Validation metrics by candidate model")

lower_better = {"brier_score", "log_loss", "mse", "mae", "rmse"}
if primary in lower_better:
    best_model = min(records, key=lambda n: records[n][0][primary])
else:
    best_model = max(records, key=lambda n: records[n][0][primary])
best_metrics = records[best_model][0]
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
auc_key = "roc_auc" if "roc_auc" in best_metrics else ("roc_auc_ovr" if "roc_auc_ovr" in best_metrics else None)
if auc_key:
    mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_metrics[auc_key]), source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("features", len(feature_cols), "primary", primary)
print("best_model", best_model, sig4(best_metrics[primary]))
