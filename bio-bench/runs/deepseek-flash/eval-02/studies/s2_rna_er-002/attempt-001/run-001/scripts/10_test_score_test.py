import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
ec = protocol.get("excluded_columns")
if isinstance(ec, dict):
    excluded = set(ec.get("columns") or ec.get("examples") or [])
elif isinstance(ec, list):
    excluded = set(ec)
else:
    excluded = set()
task = protocol["task"]
primary_metric = task["primary_metric"]
task_type = task.get("type", "binary")
group_col = protocol.get("group_column")
pos_decl = task.get("positive_label")

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model_a = load_pkl(inputs["07_model_train_logistic_GBE8KD6W/model.pkl"])
model_b = load_pkl(inputs["08_model_train_alternative_73DHGKGK/alternative_model.pkl"])
train = pd.read_parquet(inputs["train"]); val = pd.read_parquet(inputs["val"]); test = pd.read_parquet(inputs["test"])

def feat_cols(model, frame):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for s in model.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return [c for c in frame.columns if c not in targets and c not in excluded and c != group_col]

def val_score(model):
    cols = feat_cols(model, val)
    Xv = val[cols]
    proba = model.predict_proba(Xv); classes = list(model.classes_); yv = val[targets[0]].values
    if task_type == "binary":
        pos = pos_decl if pos_decl in classes else classes[1]
        return roc_auc_score((yv == pos).astype(int), proba[:, classes.index(pos)])
    return roc_auc_score(yv, proba, multi_class="ovr", average="macro", labels=classes)

try:
    sa, sb = val_score(model_a), val_score(model_b)
except Exception as e:
    print("val score failed:", e); sa, sb = -np.inf, -np.inf
model = model_a if sa >= sb else model_b
print("val best", round(max(sa, sb), 4))

cols = feat_cols(model, test)
proba = model.predict_proba(test[cols]); classes = list(model.classes_)
y_true = test[targets[0]].values
pos = pos_decl if pos_decl in classes else (classes[1] if len(classes) > 1 else classes[0])
pos_idx = classes.index(pos)

threshold = 0.5
try:
    mf = json.loads(inputs["09_validate_score_validation_W80PBK3D/manifest.json"].read_text())
    for src in [mf, mf.get("metrics", {}) if isinstance(mf.get("metrics"), dict) else {}]:
        for k in ["threshold", "frozen_threshold", "decision_threshold"]:
            if k in src:
                threshold = float(src[k]); break
        else:
            continue
        break
except Exception:
    pass

y_bin = (y_true == pos).astype(int); p_pos = proba[:, pos_idx]

def metrics_from(y, p_hi, thr):
    yh = (p_hi >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out = {"sensitivity": sens, "specificity": spec}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = roc_auc_score(y, p_hi)
        out["pr_auc"] = average_precision_score(y, p_hi)
        out["brier"] = brier_score_loss(y, p_hi)
        out["balanced_accuracy"] = balanced_accuracy_score(y, yh)
    else:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = out["balanced_accuracy"] = np.nan
    return out

order = ["roc_auc", "pr_auc", "balanced_accuracy", "brier", "sensitivity", "specificity"]
point = metrics_from(y_bin, p_pos, threshold)
rng = np.random.RandomState(42); n = len(y_bin)
boot = {k: [] for k in order}
for _ in range(1000):
    idx = rng.randint(0, n, n)
    try:
        m = metrics_from(y_bin[idx], p_pos[idx], threshold)
        for k, v in m.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                boot[k].append(float(v))
    except Exception:
        pass

rows = []
for k in order:
    v = point.get(k, np.nan); arr = boot.get(k, [])
    lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if arr else (np.nan, np.nan)
    rows.append({"metric": k, "value": round(float(v), 6) if not np.isnan(v) else np.nan,
                 "ci_low": round(lo, 6) if not np.isnan(lo) else np.nan,
                 "ci_high": round(hi, 6) if not np.isnan(hi) else np.nan})
tm = pd.DataFrame(rows)
mlca_runtime.save_table(tm, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pr = tm[tm["metric"] == primary_metric]
if len(pr):
    v = float(pr["value"].iloc[0]); lo = float(pr["ci_low"].iloc[0]); hi = float(pr["ci_high"].iloc[0])
    mlca_runtime.record_metric(primary_metric, v, "test", ci=[lo, hi])
    if primary_metric == "roc_auc":
        mlca_runtime.record_fact_candidate("test_roc_auc", v, unit="roc_auc", source="test_metrics.csv")

if task_type == "binary":
    y_pred = np.where(p_pos >= threshold, classes[pos_idx], [c for c in classes if c != pos][0])
else:
    y_pred = np.array(classes)[np.argmax(proba, axis=1)]
pred = pd.DataFrame({"row_id": test.index, "partition": "test",
                     f"y_true__{targets[0]}": y_true, f"y_pred__{targets[0]}": y_pred})
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")
print("test rows", len(pred), "primary", primary_metric, "=", round(v, 4) if len(pr) else None)
