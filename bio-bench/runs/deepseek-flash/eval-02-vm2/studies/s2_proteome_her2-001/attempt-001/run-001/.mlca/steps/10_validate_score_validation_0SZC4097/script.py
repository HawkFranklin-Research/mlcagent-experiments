import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = protocol.get("excluded_columns", [])
if isinstance(excluded, dict):
    excluded = excluded.get("columns") or excluded.get("examples") or []
excluded = list(excluded)
group = protocol.get("group_column")
target_col = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        nm = str(s.name).lower()
        return ("id" in nm or "index" in nm) and s.nunique(dropna=True) == len(s)
    return False

excl_set = set(targets) | set(excluded) | ({group} if group else set())
feature_columns = [c for c in train.columns if c not in excl_set and not is_key_like(train[c])]

def sig4(x):
    try:
        x = float(x)
        return float(f"{x:.4g}") if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def get_classes(m):
    c = getattr(m, "classes_", None)
    if c is None and hasattr(m, "steps"):
        c = m.steps[-1][1].classes_
    return np.asarray(c)

cand_specs = [
    ("08_logistic", "08_model_train_logistic_9HTWE67Q/model.pkl"),
    ("09_alternative", "09_model_train_alternative_K2EW2B11/alternative_model.pkl"),
    ("latest", "latest_model"),
]
seen, cands = set(), []
for name, key in cand_specs:
    if key not in inp:
        continue
    p = inp[key]
    rp = p.resolve()
    if rp in seen:
        continue
    seen.add(rp)
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "best_estimator_") and not hasattr(obj, "predict_proba"):
        obj = obj.best_estimator_
    if not hasattr(obj, "predict_proba"):
        raise TypeError(f"{name} has no predict_proba")
    cands.append((name, obj))

pos_label = task.get("positive_label", task.get("positive_class"))
rows, results = [], []
for name, model in cands:
    names = getattr(model, "feature_names_in_", None)
    names = list(names) if names is not None else feature_columns
    missing = [c for c in names if c not in val.columns]
    if missing:
        raise KeyError(f"{name} missing validation columns: {missing[:3]}")
    bad = set(names) & excl_set
    if bad:
        raise ValueError(f"{name} uses excluded/target columns: {sorted(bad)[:3]}")
    X = val.loc[:, names]
    proba = model.predict_proba(X)
    classes = get_classes(model)
    y = val[target_col].to_numpy()
    primary_name = task["primary_metric"]

    if task.get("type") == "binary" or len(classes) == 2:
        pl = pos_label if pos_label is not None else classes[1]
        if not np.any(classes == pl):
            raise ValueError(f"{name} positive label {pl} not in {classes}")
        idx = int(np.where(classes == pl)[0][0])
        p = proba[:, idx]
        yb = (y == pl).astype(int)
        roc = roc_auc_score(yb, p) if len(np.unique(yb)) > 1 else np.nan
        pr = average_precision_score(yb, p) if len(np.unique(yb)) > 1 else np.nan
        fpr, tpr, thr = roc_curve(yb, p)
        j = tpr - fpr
        ti = int(np.argmax(j))
        threshold = float(thr[ti]) if np.isfinite(thr[ti]) else 0.5
        bal = balanced_accuracy_score(yb, (p >= threshold).astype(int))
        brier = brier_score_loss(yb, p)
        vals = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier_score": brier}
        primary = vals.get(primary_name, np.nan)
        if not np.isfinite(primary) and primary_name in ("average_precision", "pr_auc"):
            primary = pr
        if not np.isfinite(primary) and primary_name in ("roc_auc", "roc_auc_score"):
            primary = roc
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]:
            rows.append({"model": name, "metric": metric, "value": sig4(vals[metric]),
                         "threshold": sig4(threshold) if metric == "balanced_accuracy" else np.nan})
        if primary_name not in vals:
            rows.append({"model": name, "metric": primary_name, "value": sig4(primary), "threshold": np.nan})
        results.append({"model": name, "primary": primary, "roc_auc": roc})
    else:
        present = [c for c in classes if np.any(y == c)]
        if len(present) == len(classes):
            auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            aucs = [roc_auc_score((y == c).astype(int), proba[:, j])
                    for j, c in enumerate(classes)
                    if c in present and len(np.unique((y == c).astype(int))) > 1]
            auc = float(np.mean(aucs)) if aucs else np.nan
        pred = model.predict(X)
        f1 = f1_score(y, pred, average="macro", labels=classes)
        vals = {"roc_auc_ovr": auc, "f1_macro": f1}
        primary = vals.get(primary_name, auc if "auc" in primary_name else np.nan)
        for metric, value in vals.items():
            rows.append({"model": name, "metric": metric, "value": sig4(value), "threshold": np.nan})
        if primary_name not in vals:
            rows.append({"model": name, "metric": primary_name, "value": sig4(primary), "threshold": np.nan})
        results.append({"model": name, "primary": primary, "roc_auc": auc})

lower_better = task["primary_metric"].lower() in {"brier_score", "brier", "log_loss"}
valid = [r for r in results if np.isfinite(r["primary"])]
if not valid:
    raise ValueError("No finite primary validation metric")
ordered = sorted(valid, key=lambda r: r["primary"], reverse=lower_better)
for r in ordered:
    mlca_runtime.record_metric(task["primary_metric"], float(r["primary"]), "val")
best = ordered[-1]

mlca_runtime.save_table(pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"]),
                        "validation_metrics.csv",
                        caption="Validation metrics for candidate models with validation-only Youden threshold.")
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["roc_auc"]), source="validation_metrics.csv")

print(f"val rows={len(val)} candidates={len(results)}")
print(f"best_model={best['model']} {task['primary_metric']}={best['primary']:.4g} roc_auc={best['roc_auc']:.4g}")
