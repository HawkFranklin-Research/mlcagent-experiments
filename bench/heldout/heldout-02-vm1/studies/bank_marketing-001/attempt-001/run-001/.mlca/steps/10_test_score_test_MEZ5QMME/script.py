import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(d)}")
    return d[key]

def find_key(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names:
                return v
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None

protocol = json.loads(inputs["_protocol"].read_text())
targets = req(protocol, "targets")
excluded = set(req(protocol, "excluded_columns"))
group = protocol.get("group_column", None)
task = req(protocol, "task")
primary = req(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
for c in list(candidate_cols):
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (
        s.dtype == object or pd.api.types.is_integer_dtype(s.dtype)
    ):
        candidate_cols.remove(c)

vm_manifest = json.loads(inputs["09_validate_score_validation_N92A9E57/manifest.json"].read_text())
sel = find_key(vm_manifest, {"selected_model", "selected_candidate", "best_model",
                            "selected_path", "model_path", "selected"})
if isinstance(sel, dict):
    sel_name = str(sel.get("name") or sel.get("path") or sel.get("model") or sel)
else:
    sel_name = str(sel) if sel is not None else ""

if "07" in sel_name or "logistic" in sel_name.lower():
    model_path = inputs["07_model_train_logistic_GZE7Y5E0/model.pkl"]
else:
    model_path = inputs["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

feature_cols = list(getattr(model, "feature_names_in_", candidate_cols))
bad = set(feature_cols) & (set(targets) | excluded)
if bad:
    raise ValueError(f"Feature list contains target/excluded columns: {sorted(bad)}")

target = targets[0]
y_true = test[target]
X_test = test[feature_cols].copy()
proba = model.predict_proba(X_test)
classes = np.asarray(getattr(model, "classes_", []))
if classes.size == 0:
    for step in reversed(list(model.named_steps.values())):
        if hasattr(step, "classes_"):
            classes = np.asarray(step.classes_)
            break
if classes.size != 2:
    raise ValueError("This script expects a binary task with two model classes.")

pos_label = task.get("positive_label", classes[1])
if pos_label not in classes:
    raise ValueError(f"Positive label {pos_label!r} not in model classes {list(classes)}")
pos_idx = int(np.where(classes == pos_label)[0][0])
neg_label = classes[1 - pos_idx]
p_pos = proba[:, pos_idx]

thr = find_key(vm_manifest, {"threshold", "decision_threshold", "best_threshold", "threshold_value"})
thr = float(thr) if thr is not None else 0.5
y_pred = np.where(p_pos >= thr, pos_label, neg_label)

def all_metrics(y, p, pred):
    pos = (np.asarray(y) == pos_label).astype(int)
    out = {}
    out["roc_auc"] = roc_auc_score(pos, p) if len(np.unique(pos)) > 1 else np.nan
    out["pr_auc"] = average_precision_score(pos, p) if len(np.unique(pos)) > 1 else np.nan
    out["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    out["brier_score"] = brier_score_loss(pos, p)
    cm = confusion_matrix(y, pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return out

values = all_metrics(y_true, p_pos, y_pred)
metric_names = [primary] + [k for k in values if k != primary]
rng = np.random.default_rng(42)
boot = {k: [] for k in metric_names}
for _ in range(1000):
    idx = rng.integers(0, len(test), len(test))
    bm = all_metrics(y_true.iloc[idx], p_pos[idx], y_pred[idx])
    for k in metric_names:
        boot[k].append(bm[k])

rows = []
for k in metric_names:
    arr = np.asarray(boot[k], dtype=float)
    ci = np.nanpercentile(arr, [2.5, 97.5]) if np.any(np.isfinite(arr)) else [np.nan, np.nan]
    rows.append({"metric": k, "value": values[k], "ci_low": ci[0], "ci_high": ci[1]})
metrics_df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv")

pred_df = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": y_true.to_numpy(),
    "y_pred": y_pred,
})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

for r in metrics_df.itertuples(index=False):
    mlca_runtime.record_metric(r.metric, float(r.value), "test",
                               ci=(float(r.ci_low), float(r.ci_high)))
mlca_runtime.record_fact_candidate("test_roc_auc", float(values["roc_auc"]),
                                   source="test_metrics.csv")

print(f"model={model_path.name}, threshold={thr:.4f}, positive={pos_label}")
print(f"test_rows={len(test)}, features={len(feature_cols)}")
print(f"{primary}={values[primary]:.4f} CI=({metrics_df.loc[metrics_df.metric == primary, 'ci_low'].iloc[0]:.4f},"
      f"{metrics_df.loc[metrics_df.metric == primary, 'ci_high'].iloc[0]:.4f})")
