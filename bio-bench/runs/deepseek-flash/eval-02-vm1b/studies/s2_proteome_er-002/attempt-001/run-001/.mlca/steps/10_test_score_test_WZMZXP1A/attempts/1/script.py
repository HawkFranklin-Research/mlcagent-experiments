import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, recall_score)

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol.get("group_column")
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(protocol.keys())}")

primary = task["primary_metric"]
target = targets[0]
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

def classes_of(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        for s in reversed(list(m.named_steps.values())):
            if hasattr(s, "classes_"):
                return np.asarray(s.classes_)
    raise ValueError("No classes_ found")

def features_of(m, df):
    names = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else None
    if names is None and hasattr(m, "named_steps"):
        for s in m.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                names = list(s.feature_names_in_)
                break
    if names is None:
        drop = set(targets) | set(excluded)
        if group_col is not None:
            drop.add(group_col)
        names = [c for c in df.columns if c not in drop]
    return df[names]

def pos_proba(m, X, cls, pos):
    return m.predict_proba(X)[:, list(cls).index(pos)]

cands = []
for path in [inp["07_model_train_logistic_RQ5H03SW/model.pkl"],
             inp["08_model_train_alternative_7HJK4116/alternative_model.pkl"]]:
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        cls = classes_of(m)
        pos = task.get("positive_label", cls[1])
        yv = val[target].values
        yb = (yv == pos).astype(int)
        p = pos_proba(m, features_of(m, val), cls, pos)
        score = roc_auc_score(yb, p) if primary != "pr_auc" else average_precision_score(yb, p)
        cands.append((score, m, list(cls), pos, path))
    except Exception as e:
        print("candidate failed", path.name, type(e).__name__)

if not cands:
    raise RuntimeError("No candidate model loaded")
lower = any(k in primary.lower() for k in ["brier", "loss", "mse", "mae"])
cands.sort(key=lambda t: t[0], reverse=not lower)
_, model, classes, pos_label, model_path = cands[0]
neg_label = classes[1 - classes.index(pos_label)]
X_val = features_of(model, val)
X_test = features_of(model, test)
proba = model.predict_proba(X_test)
p_test = proba[:, classes.index(pos_label)]

def get_threshold():
    try:
        vm = pd.read_csv(inp["09_validate_score_validation_E3F3YCMG/validation_metrics.csv"])
        for col in vm.columns:
            if "threshold" in str(col).lower():
                vals = pd.to_numeric(vm[col], errors="coerce").dropna()
                if len(vals):
                    return float(vals.iloc[0])
        if "metric" in vm.columns and "value" in vm.columns:
            mask = vm["metric"].astype(str).str.contains("threshold", case=False, na=False)
            vals = pd.to_numeric(vm.loc[mask, "value"], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    except Exception:
        pass
    pv = pos_proba(model, X_val, classes, pos_label)
    yv = (val[target].values == pos_label).astype(int)
    best_t, best_s = 0.5, -1.0
    for t in np.unique(pv):
        s = balanced_accuracy_score(yv, (pv >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t

threshold = get_threshold()
y_true = test[target].values
y_pred = np.where(p_test >= threshold, pos_label, neg_label)

def calc_metrics(y, yp, p):
    yb = (y == pos_label).astype(int)
    out = {}
    try: out["roc_auc"] = roc_auc_score(yb, p)
    except Exception: out["roc_auc"] = np.nan
    try: out["pr_auc"] = average_precision_score(yb, p)
    except Exception: out["pr_auc"] = np.nan
    try: out["balanced_accuracy"] = balanced_accuracy_score(y, yp)
    except Exception: out["balanced_accuracy"] = np.nan
    try: out["brier_score"] = brier_score_loss(yb, p)
    except Exception: out["brier_score"] = np.nan
    try: out["sensitivity"] = recall_score(y, yp, pos_label=pos_label, zero_division=0)
    except Exception: out["sensitivity"] = np.nan
    try: out["specificity"] = recall_score(y, yp, pos_label=neg_label, zero_division=0)
    except Exception: out["specificity"] = np.nan
    return out

base = calc_metrics(y_true, y_pred, p_test)
if primary not in base:
    base[primary] = base["roc_auc"]
rng = np.random.default_rng(SEED)
n = len(y_true)
boot = {k: [] for k in base}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    mm = calc_metrics(y_true[idx], y_pred[idx], p_test[idx])
    for k in boot:
        v = mm.get(k, np.nan)
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for k, v in base.items():
    vals = boot.get(k, [])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    rows.append({"metric": k, "value": float(v), "ci_low": float(lo), "ci_high": float(hi)})
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

mrow = metrics_df.loc[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(mrow["value"]), "test",
                           (float(mrow["ci_low"]), float(mrow["ci_high"])))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(mrow["value"]), source="test_metrics.csv")

pred = pd.DataFrame({"row_id": test.index, "partition": "test",
                     f"y_true__{target}": y_true, f"y_pred__{target}": y_pred})
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print("selected", model_path.name, "threshold", round(threshold, 4))
print("test rows", len(test), "primary", primary, round(float(mrow["value"]), 4))
