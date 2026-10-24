import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score, f1_score
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
task_type = task.get("type", "binary")
target = targets[0]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_obj = load_pkl(inputs["latest_model"])
model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj

test = pd.read_parquet(inputs["test"])
if hasattr(model, "feature_names_in_"):
    feat = list(model.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feat = [c for c in test.columns if c not in drop]
X_test = test[feat]
y_true = test[target]
proba = model.predict_proba(X_test)
classes = list(model.classes_)
y_pred = model.predict(X_test)

def find_key(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names and isinstance(v, (int, float)):
                return float(v)
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None

thr_names = {"threshold", "decision_threshold", "best_threshold", "val_threshold", "validation_threshold"}
threshold = None
if isinstance(model_obj, dict) and isinstance(model_obj.get("threshold"), (int, float)):
    threshold = float(model_obj["threshold"])
for key in ["09_validate_score_validation_8V5GD8WC/manifest.json"]:
    if threshold is None and key in inputs:
        try:
            threshold = find_key(json.loads(inputs[key].read_text()), thr_names)
        except Exception:
            pass
if threshold is None and "09_validate_score_validation_8V5GD8WC/validation_metrics.csv" in inputs:
    try:
        vm = pd.read_csv(inputs["09_validate_score_validation_8V5GD8WC/validation_metrics.csv"])
        for _, row in vm.iterrows():
            s = " ".join(str(row.get(c, "")) for c in vm.columns).lower()
            if "threshold" in s:
                for c in vm.columns:
                    if "threshold" in str(c).lower() and isinstance(row[c], (int, float)):
                        threshold = float(row[c])
                        break
                if threshold is not None:
                    break
    except Exception:
        pass
if threshold is None:
    threshold = 0.5

if task_type == "binary" and len(classes) == 2:
    pos = task.get("positive_label", classes[1])
    if pos not in classes:
        pos = classes[1]
    neg = classes[0] if pos == classes[1] else classes[1]
    jpos = classes.index(pos)
    y_pred_thr = np.where(proba[:, jpos] >= threshold, pos, neg)
else:
    pos = None
    y_pred_thr = y_pred

def compute(y_t, y_p, y_prob):
    out = {}
    if task_type == "binary" and len(classes) == 2:
        j = classes.index(pos)
        yb = (y_t == pos).astype(int)
        pp = y_prob[:, j]
        out["roc_auc"] = roc_auc_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
        out["pr_auc"] = average_precision_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
        out["balanced_accuracy"] = balanced_accuracy_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
        out["brier_score"] = brier_score_loss(yb, pp)
        out["sensitivity"] = recall_score(y_t, y_p, pos_label=pos, zero_division=0)
        neg_l = classes[0] if pos == classes[1] else classes[1]
        out["specificity"] = recall_score(y_t, y_p, pos_label=neg_l, zero_division=0)
    else:
        aucs = []
        for j, c in enumerate(classes):
            yb = (y_t == c).astype(int)
            if len(np.unique(yb)) > 1:
                aucs.append(roc_auc_score(yb, y_prob[:, j]))
        out["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else np.nan
        out["f1_macro"] = f1_score(y_t, y_p, average="macro", zero_division=0) if len(np.unique(y_t)) > 1 else np.nan
    return out

metrics = compute(y_true.values, y_pred_thr, proba)
rng = np.random.default_rng(42)
n = len(test)
boot = {k: [] for k in metrics}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    try:
        m = compute(y_true.values[idx], y_pred_thr[idx], proba[idx])
        for k, v in m.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                boot[k].append(v)
    except Exception:
        pass

rows = []
for k, v in metrics.items():
    vals = boot.get(k, [])
    if vals:
        lo, hi = np.percentile(vals, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})

dfm = pd.DataFrame(rows)
if primary in dfm["metric"].values:
    dfm = pd.concat([dfm[dfm.metric == primary], dfm[dfm.metric != primary]], ignore_index=True)
for c in ["value", "ci_low", "ci_high"]:
    dfm[c] = dfm[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x)
mlca_runtime.save_table(dfm, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

for _, r in dfm.iterrows():
    ci = (r["ci_low"], r["ci_high"]) if pd.notna(r["ci_low"]) and pd.notna(r["ci_high"]) else None
    mlca_runtime.record_metric(r["metric"], float(r["value"]), "test", ci=ci)
if primary == "roc_auc" and "roc_auc" in metrics:
    mlca_runtime.record_fact_candidate("test_roc_auc", float(metrics["roc_auc"]), source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index, "partition": "test"})
out[f"y_true__{target}"] = y_true.values
out[f"y_pred__{target}"] = y_pred_thr
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print(f"test rows: {n}")
print(f"primary {primary}: {metrics.get(primary, float('nan')):.4f}")
print(f"threshold: {threshold}")
print("saved test_metrics.csv and test_predictions.parquet")
