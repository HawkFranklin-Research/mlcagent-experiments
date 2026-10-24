import json
import pickle

import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

try:
    task = protocol["task"]
    primary = task["primary_metric"]
    targets = protocol["targets"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; top-level keys: {list(protocol.keys())}") from exc

if task.get("type") != "multiclass" or primary != "roc_auc_ovr":
    raise ValueError(f"Expected multiclass roc_auc_ovr, got {task.get('type')} and {primary}")
if len(targets) != 1:
    raise ValueError(f"Expected one declared target, got {targets}")

target = targets[0]
val = pd.read_parquet(inp["val"])
if target not in val.columns:
    raise KeyError(f"Target {target!r} not in validation columns")
y = val[target]


def raw_input_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise AttributeError(f"{type(model).__name__} has no feature_names_in_")


def classes_of(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        return np.asarray(list(model.named_steps.values())[-1].classes_)
    raise AttributeError(f"{type(model).__name__} has no classes_")


def macro_ovr_auc(y_true, proba, classes):
    present = [c for c in classes if np.any(y_true == c)]
    omitted = [c for c in classes if c not in present]
    if omitted:
        print(f"classes omitted: {omitted}")
    if len(present) == len(classes):
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
    values = []
    for cls in present:
        j = int(np.where(classes == cls)[0][0])
        values.append(roc_auc_score((y_true == cls).astype(int), proba[:, j]))
    return float(np.mean(values))


candidates = [
    ("logistic", inp["08_model_train_logistic_7JTSZ4DZ/model.pkl"]),
    ("alternative", inp["09_model_train_alternative_94QWT4VC/alternative_model.pkl"]),
]
rows = []
scores = []

for name, path in candidates:
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    cols = raw_input_columns(model)
    missing = [col for col in cols if col not in val.columns]
    if missing:
        raise KeyError(f"{name} missing input columns: {missing[:5]} ({len(missing)} total)")
    X = val.loc[:, cols]
    classes = classes_of(model)
    proba = np.asarray(model.predict_proba(X))
    auc = macro_ovr_auc(y, proba, classes)
    pred = model.predict(X)
    f1 = float(f1_score(y, pred, labels=classes, average="macro", zero_division=0))
    rows.append({"model": name, "metric": primary, "value": float(f"{auc:.4g}"), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f"{f1:.4g}"), "threshold": np.nan})
    scores.append((auc, name))
    mlca_runtime.record_metric(primary, float(auc), "val")

best_score, best_name = max(scores, key=lambda item: item[0])
mlca_runtime.record_metric(primary, float(best_score), "val")
metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics by candidate")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_score), source="validation_metrics.csv")

print(f"val rows: {len(val)}, candidates: {len(candidates)}")
print(f"best: {best_name} {primary}={best_score:.4g}")
print("wrote validation_metrics.csv")
