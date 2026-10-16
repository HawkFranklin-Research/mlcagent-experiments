import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]
primary_metric = task["primary_metric"]

test = pd.read_parquet(inputs["test"])
model = pickle.load(open(inputs["latest_model"], "rb"))

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
elif hasattr(model, "steps"):
    for _, step in model.steps:
        if hasattr(step, "feature_names_in_"):
            feature_cols = list(step.feature_names_in_)
            break
    else:
        raise ValueError("No feature_names_in_ found in pipeline")
else:
    raise ValueError("Model has no feature_names_in_")

for col in targets + excluded + ([group] if group else []):
    if col in feature_cols:
        raise ValueError(f"Column {col} must not be a feature")

X_test = test[feature_cols].copy()
y_test = test[targets[0]]

if hasattr(model, "classes_"):
    classes = model.classes_
elif hasattr(model, "steps"):
    classes = model.steps[-1][1].classes_
else:
    raise ValueError("Cannot find classes_")

pos_label = protocol.get("positive_label") or task.get("positive_label") or classes[1]
pos_idx = list(classes).index(pos_label)
y_true_bin = (y_test == pos_label).astype(int)
proba_pos = model.predict_proba(X_test)[:, pos_idx]

def find_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "threshold" in k.lower() and isinstance(v, (int, float)):
                return v
            res = find_threshold(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_threshold(item)
            if res is not None:
                return res
    return None

threshold = find_threshold(json.loads(inputs["09_validate_score_validation_E0QH2BPW/manifest.json"].read_text()))
if threshold is None:
    vm = pd.read_csv(inputs["09_validate_score_validation_E0QH2BPW/validation_metrics.csv"])
    if "metric" in vm.columns and "value" in vm.columns:
        row = vm[vm["metric"].str.contains("threshold", case=False, na=False)]
        if not row.empty:
            threshold = float(row.iloc[0]["value"])
    elif "threshold" in vm.columns:
        threshold = float(vm["threshold"].iloc[0])

if threshold is None:
    val = pd.read_parquet(inputs["val"])
    X_val = val[feature_cols].copy()
    y_val_bin = (val[targets[0]] == pos_label).astype(int)
    proba_val = model.predict_proba(X_val)[:, pos_idx]
    best_t, best_s = 0.5, -1
    for t in np.linspace(0, 1, 101):
        yp = (proba_val >= t).astype(int)
        if len(np.unique(yp)) < 2:
            continue
        s = balanced_accuracy_score(y_val_bin, yp)
        if s > best_s:
            best_s, best_t = s, t
    threshold = best_t

def compute_metrics(y_true, proba, thresh):
    y_pred = (proba >= thresh).astype(int)
    res = {}
    try: res["roc_auc"] = roc_auc_score(y_true, proba)
    except ValueError: res["roc_auc"] = np.nan
    try: res["pr_auc"] = average_precision_score(y_true, proba)
    except ValueError: res["pr_auc"] = np.nan
    try: res["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    except ValueError: res["balanced_accuracy"] = np.nan
    try: res["brier_score"] = brier_score_loss(y_true, proba)
    except ValueError: res["brier_score"] = np.nan
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        res["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        res["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except ValueError:
        res["sensitivity"] = np.nan
        res["specificity"] = np.nan
    return res

metrics = compute_metrics(y_true_bin, proba_pos, threshold)
rng = np.random.default_rng(42)
n = len(y_true_bin)
boot = {k: [] for k in metrics}
for _ in range(1000):
    idx = rng.choice(n, n, replace=True)
    m = compute_metrics(y_true_bin.iloc[idx], proba_pos[idx], threshold)
    for k, v in m.items():
        boot[k].append(v)

rows = []
for k, v in metrics.items():
    arr = np.array(boot[k])
    arr = arr[~np.isnan(arr)]
    ci_low = np.percentile(arr, 2.5) if len(arr) else np.nan
    ci_high = np.percentile(arr, 97.5) if len(arr) else np.nan
    rows.append({"metric": k, "value": v, "ci_low": ci_low, "ci_high": ci_high})

df_metrics = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df_metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

primary_val = metrics[primary_metric]
primary_ci = (df_metrics.loc[df_metrics["metric"] == primary_metric, "ci_low"].values[0],
              df_metrics.loc[df_metrics["metric"] == primary_metric, "ci_high"].values[0])
mlca_runtime.record_metric(primary_metric, primary_val, split="test", ci=primary_ci)
if primary_metric == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", primary_val, source="test_metrics.csv")

print(f"Test rows: {len(test)}")
print(f"Features: {len(feature_cols)}")
print(f"Threshold: {threshold:.4f}")
print(f"{primary_metric}: {primary_val:.4f} (95% CI: {primary_ci[0]:.4f}-{primary_ci[1]:.4f})")
