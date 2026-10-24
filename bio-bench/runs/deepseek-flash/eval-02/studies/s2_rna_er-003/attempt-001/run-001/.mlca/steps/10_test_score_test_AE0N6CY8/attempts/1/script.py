import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
)

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available task keys: {list(task)}")
primary = task["primary_metric"]
targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

test = pd.read_parquet(inp["test"])
model_path = inp.get("latest_model") or inp["07_model_train_logistic_6PX29S2M/model.pkl"]
with open(model_path, "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict):
    for k in ("model", "estimator", "pipeline"):
        if k in model:
            model = model[k]
            break

classes = list(getattr(model, "classes_", []))
if not classes and hasattr(model, "named_steps"):
    last = list(model.named_steps.values())[-1]
    classes = list(getattr(last, "classes_", []))
if len(classes) != 2:
    raise ValueError(f"Expected binary model classes, got {classes}")

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

cols = feature_names(model)
if cols is None:
    cols = []
    for c in test.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = test[c]
        if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
            continue
        cols.append(c)
missing = [c for c in cols if c not in test.columns]
if missing:
    raise KeyError(f"Missing model feature columns: {missing[:5]}")
X = test.loc[:, cols]
target = targets[0]
y_true = test[target].copy()
proba = np.asarray(model.predict_proba(X))

threshold = None
for attr in dir(model):
    if "threshold" in attr.lower():
        try:
            v = getattr(model, attr)
        except Exception:
            continue
        if isinstance(v, (int, float, np.floating)):
            threshold = float(v)
            break

def walk_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float, np.floating)):
                return float(v)
            r = walk_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = walk_threshold(v)
            if r is not None:
                return r
    return None

if threshold is None:
    for key in (
        "09_validate_score_validation_B7X4RBAC/validation_metrics.csv",
        "09_validate_score_validation_B7X4RBAC/validation_metrics.parquet",
    ):
        p = inp.get(key)
        if p is not None and p.exists():
            try:
                vm = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
                if "metric" in vm.columns and "value" in vm.columns:
                    hit = vm[vm["metric"].astype(str).str.lower().str.contains("threshold")]
                    if len(hit):
                        threshold = float(hit.iloc[0]["value"])
                        break
            except Exception:
                pass
if threshold is None:
    for key in (
        "09_validate_score_validation_B7X4RBAC/manifest.json",
        "07_model_train_logistic_6PX29S2M/manifest.json",
        "08_model_train_alternative_MSPMPPWQ/manifest.json",
    ):
        p = inp.get(key)
        if p is not None and p.exists():
            try:
                r = walk_threshold(json.loads(p.read_text()))
                if r is not None:
                    threshold = r
                    break
            except Exception:
                pass
if threshold is None:
    threshold = 0.5

pos = task.get("positive_label", protocol.get("positive_label"))
if pos is None:
    pos = classes[1]
pos_idx = classes.index(pos)
p1 = proba[:, pos_idx]
ybin = (y_true == pos).astype(int).to_numpy()
predbin = (p1 >= threshold).astype(int).to_numpy()

def compute_metrics(y, p, pred):
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y, p)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y, p)
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    except Exception:
        out["balanced_accuracy"] = np.nan
    out["brier_score"] = brier_score_loss(y, p)
    out["sensitivity"] = recall_score(y, pred, pos_label=1, zero_division=0)
    out["specificity"] = recall_score(y, pred, pos_label=0, zero_division=0)
    return out

point = compute_metrics(ybin, p1, predbin)
if primary not in point:
    raise ValueError(f"Unsupported primary_metric {primary!r}; computed: {list(point)}")

rng = np.random.default_rng(SEED)
boot = {k: [] for k in point}
n = len(ybin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    mb = compute_metrics(ybin[idx], p1[idx], predbin[idx])
    for k, v in mb.items():
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for name, value in point.items():
    vals = np.asarray(boot[name])
    lo = float(np.percentile(vals, 2.5)) if len(vals) else np.nan
    hi = float(np.percentile(vals, 97.5)) if len(vals) else np.nan
    rows.append({"metric": name, "value": float(value), "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows)
order = [primary] + [m for m in metrics_df["metric"] if m != primary]
metrics_df = metrics_df.set_index("metric").loc[order].reset_index()
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

prow = metrics_df[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prow["value"]), "test", (float(prow["ci_low"]), float(prow["ci_high"])))
if "roc_auc" in set(metrics_df["metric"]):
    r = metrics_df[metrics_df["metric"] == "roc_auc"].iloc[0]
    mlca_runtime.record_fact_candidate("test_roc_auc", float(r["value"]), source="test_metrics.csv")

neg = [c for c in classes if c != pos][0]
y_pred = np.where(predbin == 1, pos, neg)
pred_df = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": y_true.to_numpy(),
    "y_pred": y_pred,
})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df.reset_index(drop=True), "test_predictions.parquet")

print(f"test rows={len(test)} features={len(cols)} classes={classes}")
print(f"threshold={threshold:.6g} positive_label={pos!r}")
print(f"{primary}={float(prow['value']):.6g} CI=({float(prow['ci_low']):.6g}, {float(prow['ci_high']):.6g})")
