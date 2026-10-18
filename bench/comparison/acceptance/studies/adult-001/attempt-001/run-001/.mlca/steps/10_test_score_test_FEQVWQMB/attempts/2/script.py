import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available {list(d)}")
    return d[k]

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

task = need(protocol, "task")
primary = need(task, "primary_metric")
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = protocol.get("group_column")
target = targets[0]

man9 = json.loads(inp["09_validate_score_validation_XPS3V6ZS/manifest.json"].read_text())
sel = thr = None
for k, v in walk(man9):
    lk = k.lower()
    if thr is None and "threshold" in lk and isinstance(v, (int, float)) and not isinstance(v, bool):
        thr = float(v)
    if sel is None and (("select" in lk and ("model" in lk or "candidate" in lk)) or lk in ("best_model", "chosen_model")):
        sel = v

valm = pd.read_csv(inp["09_validate_score_validation_XPS3V6ZS/validation_metrics.csv"])
low = {c.lower(): c for c in valm.columns}
metric_col = next((c for k, c in low.items() if k in ("metric", "metric_name")), None)
value_col = next((c for k, c in low.items() if k in ("value", "score", "metric_value")), None)
cand_col = next((c for k, c in low.items() if k in ("model", "candidate", "model_name", "candidate_name", "name")), None)

if sel is None and metric_col and value_col:
    rows = valm[valm[metric_col].astype(str).str.lower() == primary.lower()]
    if len(rows):
        higher = primary.lower() not in ("log_loss", "brier_score", "brier", "mae", "mse", "rmse")
        idx = rows[value_col].idxmax() if higher else rows[value_col].idxmin()
        if cand_col:
            sel = rows.loc[idx, cand_col]
if sel is None and primary.lower() in low:
    vals = valm[low[primary.lower()]]
    idx = vals.idxmax()
    if cand_col:
        sel = valm.loc[idx, cand_col]
if thr is None and metric_col and value_col:
    rows = valm[valm[metric_col].astype(str).str.lower().isin(["threshold", "decision_threshold", "threshold_value"])]
    if len(rows):
        thr = float(rows[value_col].iloc[0])

if isinstance(sel, dict):
    sel = sel.get("path", sel.get("name", str(sel)))
if sel is None:
    sel = "latest"
s = str(sel).lower()
if "07" in s or "logistic" in s:
    model_path = inp["07_model_train_logistic_QY4T2JSR/model.pkl"]
elif "08" in s or "alternative" in s or "latest" in s:
    model_path = inp["08_model_train_alternative_5C3RWNJ3/alternative_model.pkl"]
else:
    model_path = inp["latest_model"]
if thr is None:
    raise ValueError("Missing frozen validation threshold in 09 manifest/metrics")

model = pickle.loads(model_path.read_bytes())
if isinstance(model, dict) and "model" in model:
    model = model["model"]

def get_classes(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise ValueError("Model has no classes_")

classes = get_classes(model)
test = pd.read_parquet(inp["test"])
test = test.loc[test[target].notna()].copy()
y_true = test[target].to_numpy()

fcols = None
if hasattr(model, "feature_names_in_"):
    fcols = list(model.feature_names_in_)
elif hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            fcols = list(step.feature_names_in_)
            break
drop = set(targets) | set(excluded)
if group:
    drop.add(str(group))
if not fcols or not set(fcols).issubset(set(test.columns)):
    fcols = [c for c in test.columns if c not in drop]
X = test[fcols].copy()

pos = task.get("positive_label", task.get("positive_class", task.get("positive")))
if pos is None:
    pos = classes[1]
if pos not in classes:
    raise ValueError(f"Positive label {pos} not in model classes {classes}")
proba = np.asarray(model.predict_proba(X))
p = proba[:, int(np.where(classes == pos)[0][0])]
y = (y_true == pos).astype(int)
pred = (p >= thr).astype(int)

def score(yy, pp, pr):
    tn, fp, fn, tp = confusion_matrix(yy, pr, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
        "pr_auc": average_precision_score(yy, pp),
        "balanced_accuracy": balanced_accuracy_score(yy, pr),
        "brier_score": brier_score_loss(yy, pp),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
    }

metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
point = score(y, p, pred)
rng = np.random.default_rng(42)
boot = {m: [] for m in metric_names}
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yy, pp, pr = y[idx], p[idx], pred[idx]
    if len(np.unique(yy)) < 2:
        continue
    try:
        vals = score(yy, pp, pr)
        for m in metric_names:
            if np.isfinite(vals[m]):
                boot[m].append(vals[m])
    except Exception:
        pass

ci = {}
for m in metric_names:
    arr = np.asarray(boot[m], dtype=float)
    ci[m] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

rows = [{"metric": m, "value": point[m], "ci_low": ci[m][0], "ci_high": ci[m][1]} for m in metric_names]
df = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    df[c] = df[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) and np.isfinite(x) else np.nan)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Frozen test metrics with 95% bootstrap CI")

if primary in point:
    mlca_runtime.record_metric(primary, float(point[primary]), "test", ci=ci[primary])
else:
    mlca_runtime.record_metric(primary, float("nan"), "test", ci=(np.nan, np.nan))
mlca_runtime.record_fact_candidate("test_roc_auc", float(point["roc_auc"]), source="test_metrics.csv")
print("test rows", len(y))
print("primary", primary, point.get(primary), ci.get(primary))
