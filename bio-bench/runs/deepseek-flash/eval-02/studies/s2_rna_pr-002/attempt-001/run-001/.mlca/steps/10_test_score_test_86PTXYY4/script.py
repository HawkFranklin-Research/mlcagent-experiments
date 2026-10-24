import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score)
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}; available keys: {sorted(d)}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = set(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
task = need(protocol, "task")
primary_name = need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

def is_key_like(s):
    if len(s) <= 1:
        return False
    if pd.api.types.is_integer_dtype(s):
        return s.is_unique
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.is_unique
    return False

feature_cols = [c for c in train.columns
                if c not in excluded and c not in targets and c != group and not is_key_like(train[c])]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

expected = list(getattr(model, "feature_names_in_", feature_cols))
forbidden = set(targets) | excluded | ({group} if group is not None else set())
if set(expected) & forbidden:
    raise ValueError("Model expects a target/excluded/group column")
if any(c not in test.columns for c in expected):
    raise KeyError("Test is missing a model feature column")
X = test[expected]
proba = model.predict_proba(X)

if hasattr(model, "classes_"):
    classes = list(model.classes_)
else:
    classes = list(model.steps[-1][1].classes_)

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "threshold" in str(k).lower():
                try:
                    return float(v)
                except Exception:
                    pass
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_thr(v)
            if r is not None:
                return r
    return None

thr = None
try:
    man = json.loads(inputs["09_validate_score_validation_A3KYD098/manifest.json"].read_text())
    thr = find_thr(man)
except Exception:
    pass
if thr is None:
    vm = pd.read_csv(inputs["09_validate_score_validation_A3KYD098/validation_metrics.csv"])
    for c in vm.columns:
        if "threshold" in c.lower():
            vals = vm[c].dropna()
            if len(vals):
                thr = float(vals.iloc[0])
                break
    if thr is None and {"metric", "value"} <= set(vm.columns):
        m = vm["metric"].astype(str).str.lower()
        hit = m.str.contains("threshold")
        if hit.any():
            thr = float(vm.loc[hit, "value"].iloc[0])
if thr is None:
    thr = 0.5

def binary_metrics(yt, pr, pos_label, threshold):
    yb = (yt == pos_label).astype(int)
    j = classes.index(pos_label)
    pp = pr[:, j]
    pred = (pp >= threshold).astype(int)
    neg = [c for c in classes if c != pos_label][0]
    ypred = np.where(pred == 1, pos_label, neg)
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(yb, pp)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(yb, pp)
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["balanced_accuracy"] = balanced_accuracy_score(yt, ypred)
    except Exception:
        out["balanced_accuracy"] = np.nan
    try:
        out["brier_score"] = brier_score_loss(yb, pp)
    except Exception:
        out["brier_score"] = np.nan
    tp = np.sum((yb == 1) & (pred == 1))
    fn = np.sum((yb == 1) & (pred == 0))
    tn = np.sum((yb == 0) & (pred == 0))
    fp = np.sum((yb == 0) & (pred == 1))
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out, ypred

y_true = test[targets[0]].to_numpy()
if len(classes) == 2:
    pos = task.get("positive_label", protocol.get("positive_label"))
    if pos is None:
        pos = classes[1]
    if pos not in classes:
        raise ValueError(f"Positive label {pos!r} not in classes {classes}")
    metrics, y_pred = binary_metrics(y_true, proba, pos, thr)
else:
    y_pred = np.array(classes)[np.argmax(proba, axis=1)]
    present = [c for c in classes if c in set(y_true)]
    if len(present) == len(classes):
        roc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        roc = np.nanmean([roc_auc_score((y_true == c).astype(int), proba[:, j])
                          for j, c in enumerate(classes) if c in present])
    metrics = {primary_name: roc, "macro_f1": f1_score(y_true, y_pred, average="macro")}

n = len(test)
rng = np.random.default_rng(42)
boot = {k: [] for k in metrics}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    if len(classes) == 2:
        m, _ = binary_metrics(y_true[idx], proba[idx], pos, thr)
    else:
        yt = y_true[idx]
        present = [c for c in classes if c in set(yt)]
        if len(present) == len(classes):
            rv = roc_auc_score(yt, proba[idx], multi_class="ovr", average="macro", labels=classes)
        else:
            rv = np.nanmean([roc_auc_score((yt == c).astype(int), proba[idx][:, j])
                             for j, c in enumerate(classes) if c in present])
        m = {primary_name: rv, "macro_f1": f1_score(yt, y_pred[idx], average="macro")}
    for k, v in m.items():
        if np.isfinite(v):
            boot[k].append(v)

rows, ci_map = [], {}
for k, v in metrics.items():
    vals = boot.get(k, [])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    ci_map[k] = (lo, hi)

df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pv = float(metrics[primary_name])
plo, phi = ci_map[primary_name]
mlca_runtime.record_metric(primary_name, pv, "test", ci=(float(plo), float(phi)))
if primary_name == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", pv, source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index.to_numpy(), "partition": "test"})
out["y_true"] = y_true
out["y_pred"] = y_pred
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print(f"test rows={len(test)} features={len(expected)} threshold={thr:.4g}")
print(f"{primary_name}={pv:.4f} ci=({plo:.4f},{phi:.4f})")
