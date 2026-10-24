import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, brier_score_loss)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]; excluded = proto["excluded_columns"]
group_col = proto.get("group_column")
task = proto["task"]; primary_metric = task["primary_metric"]
assert task["type"] == "binary", "binary-only script"

train_df = pd.read_parquet(inputs["train"])
test_df = pd.read_parquet(inputs["test"])

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else \
    [c for c in train_df.columns if c not in targets and c not in excluded and c != group_col]

proba = model.predict_proba(test_df[feats])
classes = list(model.classes_)
tgt = targets[0]
y_true = test_df[tgt].values

# Frozen validation threshold: look in manifest / metrics; default 0.5
thr = 0.5
vp = inputs.get("09_validate_score_validation_EVCXVFYK/manifest.json")
if vp is not None and vp.exists():
    try:
        vj = json.loads(vp.read_text())
        for k in ("threshold", "frozen_threshold", "decision_threshold"):
            if k in vj:
                thr = float(vj[k]); break
    except Exception:
        pass
try:
    vpar = pd.read_parquet(inputs["09_validate_score_validation_EVCXVFYK/validation_metrics.parquet"])
    if "metric" in vpar.columns and "threshold" in vpar["metric"].values:
        thr = float(vpar.loc[vpar["metric"] == "threshold", "value"].iloc[0])
except Exception:
    pass

pos_label = task.get("positive_label", classes[1])
pos_idx = classes.index(pos_label)
p_pos = proba[:, pos_idx]
y_bin = (y_true == pos_label).astype(int)
y_pred = np.where(p_pos >= thr, pos_label, classes[1 - pos_idx])

def _ba(y, s):
    return balanced_accuracy_score(y, (s >= thr).astype(int))
def _sens(y, s):
    yh = (s >= thr).astype(int); d = int((y == 1).sum())
    return float(((yh == 1) & (y == 1)).sum()) / d if d else np.nan
def _spec(y, s):
    yh = (s >= thr).astype(int); d = int((y == 0).sum())
    return float(((yh == 0) & (y == 0)).sum()) / d if d else np.nan

metrics = {
    "roc_auc": lambda y, s: roc_auc_score(y, s),
    "pr_auc": lambda y, s: average_precision_score(y, s),
    "balanced_accuracy": _ba,
    "brier_score": lambda y, s: brier_score_loss(y, s),
    "sensitivity": _sens,
    "specificity": _spec,
}

rng = np.random.RandomState(42)
boot = {k: [] for k in metrics}
for _ in range(1000):
    b = rng.randint(0, len(y_bin), len(y_bin))
    if len(np.unique(y_bin[b])) < 2:
        continue
    for k, fn in metrics.items():
        try:
            v = fn(y_bin[b], p_pos[b])
            if not np.isnan(v):
                boot[k].append(float(v))
        except Exception:
            pass

rows = []
for k, fn in metrics.items():
    try:
        val = float(fn(y_bin, p_pos))
    except Exception:
        val = np.nan
    arr = np.array(boot[k])
    lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) >= 10 else (np.nan, np.nan)
    rows.append({"metric": k,
                 "value": round(val, 4) if not np.isnan(val) else np.nan,
                 "ci_low": round(lo, 4) if not np.isnan(lo) else np.nan,
                 "ci_high": round(hi, 4) if not np.isnan(hi) else np.nan})

tm = pd.DataFrame(rows)
mlca_runtime.save_table(tm, "test_metrics.csv")

pm_row = tm[tm["metric"] == primary_metric].iloc[0]
mlca_runtime.record_metric(primary_metric, float(pm_row["value"]), "test",
                           ci=[float(pm_row["ci_low"]), float(pm_row["ci_high"])])
roc_val = float(tm.loc[tm["metric"] == "roc_auc", "value"].iloc[0])
mlca_runtime.record_fact_candidate("test_roc_auc", roc_val, source="test_metrics.csv")

pred_df = pd.DataFrame({
    f"y_true__{tgt}": y_true,
    f"y_pred__{tgt}": y_pred,
    "partition": "test",
}, index=test_df.index)
for i, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, i]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

print(f"n_test={len(test_df)} n_feats={len(feats)} thr={thr:.3f}")
print(f"{primary_metric}={pm_row['value']:.4f} CI=[{pm_row['ci_low']:.4f},{pm_row['ci_high']:.4f}]")
