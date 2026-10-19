import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {sorted(protocol.keys())}")

targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]
excluded = protocol["excluded_columns"]

task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task.keys())}")
primary = task["primary_metric"]
if primary != "roc_auc_ovr":
    raise ValueError(f"Expected multiclass roc_auc_ovr, got {primary}")

test = pd.read_parquet(inputs["test"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "feature_names_in_"):
    raise AttributeError("Loaded model has no feature_names_in_")
features = list(model.feature_names_in_)
if target in features:
    raise ValueError("Target column in model features")

X = test[features]
y = test[target].to_numpy()
pred = np.asarray(model.predict(X))
proba = np.asarray(model.predict_proba(X))
classes = np.asarray(model.classes_)

def macro_ovr_auc(y_true, p):
    vals = []
    for j, c in enumerate(classes):
        yb = (y_true == c)
        if yb.any() and (~yb).any():
            vals.append(roc_auc_score(yb, p[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def macro_f1(y_true, pred_true):
    return float(f1_score(y_true, pred_true, labels=classes, average="macro", zero_division=0))

auc = macro_ovr_auc(y, proba)
f1 = macro_f1(y, pred)

rng = np.random.default_rng(42)
n = len(y)
aucs = np.empty(1000)
f1s = np.empty(1000)
for i in range(1000):
    idx = rng.integers(0, n, n)
    yb = y[idx]
    aucs[i] = macro_ovr_auc(yb, proba[idx])
    f1s[i] = macro_f1(yb, pred[idx])

auc_ci = np.nanpercentile(aucs, [2.5, 97.5])
f1_ci = np.nanpercentile(f1s, [2.5, 97.5])

def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

metrics = pd.DataFrame([
    {"metric": primary, "value": sig4(auc), "ci_low": sig4(auc_ci[0]), "ci_high": sig4(auc_ci[1])},
    {"metric": "macro_f1", "value": sig4(f1), "ci_low": sig4(f1_ci[0]), "ci_high": sig4(f1_ci[1])},
])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Frozen model test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, float(auc), "test", ci=(float(auc_ci[0]), float(auc_ci[1])))
mlca_runtime.record_metric("macro_f1", float(f1), "test", ci=(float(f1_ci[0]), float(f1_ci[1])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

present = [c for c in classes if (y == c).any()]
missing_classes = [c for c in classes if c not in present]
print(f"test rows={n}, features={len(features)}, classes={len(classes)}")
print(f"{primary}={auc:.4f} CI=({auc_ci[0]:.4f},{auc_ci[1]:.4f})")
print(f"macro_f1={f1:.4f} CI=({f1_ci[0]:.4f},{f1_ci[1]:.4f})")
if missing_classes:
    print(f"classes not in test: {missing_classes}")
