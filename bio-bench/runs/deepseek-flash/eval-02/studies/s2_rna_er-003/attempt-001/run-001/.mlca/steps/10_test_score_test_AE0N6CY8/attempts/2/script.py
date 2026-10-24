import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score,
)

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
task = protocol["task"]
primary = task.get("primary_metric")
if not primary:
    raise KeyError(f"Missing task.primary_metric; available task keys: {list(task)}")
targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")

test = pd.read_parquet(inp["test"])
target = targets[0]
y_true = test[target].copy()

model_path = inp.get("latest_model") or inp["07_model_train_logistic_6PX29S2M/model.pkl"]
with open(model_path, "rb") as f:
    model = pickle.load(f)
while isinstance(model, dict):
    for key in ("model", "estimator", "pipeline"):
        if key in model:
            model = model[key]
            break
    else:
        raise ValueError(f"Unknown dict wrapper keys: {list(model)}")

classes = list(getattr(model, "classes_", []))
if not classes and hasattr(model, "named_steps"):
    classes = list(getattr(list(model.named_steps.values())[-1], "classes_", []))
if len(classes) != 2:
    raise ValueError(f"Expected binary classes, got {classes}")

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

cols = feature_names(model)
if not cols:
    cols = [c for c in test.columns
            if c not in targets and c not in excluded and c != group
            and not (pd.api.types.is_integer_dtype(test[c])
                     and test[c].nunique(dropna=False) == len(test))]
missing = [c for c in cols if c not in test.columns]
if missing:
    raise KeyError(f"Missing model feature columns: {missing[:5]}")
X = test.loc[:, cols]
proba = np.asarray(model.predict_proba(X))

threshold = 0.5
for key in ("09_validate_score_validation_B7X4RBAC/manifest.json",
            "07_model_train_logistic_6PX29S2M/manifest.json",
            "08_model_train_alternative_MSPMPPWQ/manifest.json"):
    p = inp.get(key)
    if p is None or not p.exists():
        continue
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "threshold" in str(k).lower() and isinstance(v, (int, float)):
                    return float(v)
                r = scan(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = scan(v)
                if r is not None:
                    return r
        return None
    t = scan(data)
    if t is not None:
        threshold = t
        break

pos = task.get("positive_label")
if pos is None:
    pos = protocol.get("positive_label")
if pos is None:
    pos = classes[1]
if pos not in classes:
    raise ValueError(f"Positive label {pos!r} not in classes {classes}")
pos_idx = classes.index(pos)
p1 = proba[:, pos_idx]
ybin = (y_true == pos).astype(int).to_numpy()
predbin = (p1 >= threshold).astype(int)

def compute(y, p, pred):
    d = {}
    try: d["roc_auc"] = float(roc_auc_score(y, p))
    except Exception: d["roc_auc"] = np.nan
    try: d["pr_auc"] = float(average_precision_score(y, p))
    except Exception: d["pr_auc"] = np.nan
    try: d["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    except Exception: d["balanced_accuracy"] = np.nan
    try: d["brier_score"] = float(brier_score_loss(y, p))
    except Exception: d["brier_score"] = np.nan
    try: d["sensitivity"] = float(recall_score(y, pred, pos_label=1, zero_division=0))
    except Exception: d["sensitivity"] = np.nan
    try: d["specificity"] = float(recall_score(y, pred, pos_label=0, zero_division=0))
    except Exception: d["specificity"] = np.nan
    return d

point = compute(ybin, p1, predbin)
if primary not in point:
    raise ValueError(f"Unsupported primary_metric {primary!r}; computed: {list(point)}")

rng = np.random.default_rng(SEED)
boot = {k: [] for k in point}
n = len(ybin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    mb = compute(ybin[idx], p1[idx], predbin[idx])
    for k, v in mb.items():
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for name, val in point.items():
    arr = np.asarray(boot[name])
    lo = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
    hi = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    rows.append({"metric": name,
                 "value": float(val) if np.isfinite(val) else np.nan,
                 "ci_low": lo, "ci_high": hi})
mdf = pd.DataFrame(rows)
mdf = pd.concat([mdf[mdf["metric"] == primary],
                 mdf[mdf["metric"] != primary]], ignore_index=True)
mlca_runtime.save_table(mdf, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

prow = mdf[mdf["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prow["value"]), "test",
                           (float(prow["ci_low"]), float(prow["ci_high"])))
if "roc_auc" in set(mdf["metric"].tolist()):
    r = mdf[mdf["metric"] == "roc_auc"].iloc[0]
    mlca_runtime.record_fact_candidate("test_roc_auc", float(r["value"]),
                                       source="test_metrics.csv")

neg = [c for c in classes if c != pos][0]
y_pred = np.where(predbin == 1, pos, neg)
pred_df = pd.DataFrame({
    "row_id": np.asarray(test.index),
    "partition": "test",
    "y_true": np.asarray(y_true),
    "y_pred": y_pred,
})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

print(f"test n={len(test)} features={len(cols)}")
print(f"threshold={threshold:.6g} pos={pos!r}")
print(f"{primary}={float(prow['value']):.4g} "
      f"CI=({float(prow['ci_low']):.4g},{float(prow['ci_high']):.4g})")
