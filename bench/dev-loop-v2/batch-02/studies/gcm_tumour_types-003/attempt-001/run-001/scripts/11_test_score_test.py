import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
target = targets[0]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")


def find_metric(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)) and not isinstance(obj[key], bool):
            return float(obj[key])
        for v in obj.values():
            r = find_metric(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_metric(v, key)
            if r is not None:
                return r
    return None


candidate_keys = [
    ("08", "08_model_train_logistic_SXT0E7SD/model.pkl", "08_model_train_logistic_SXT0E7SD/manifest.json"),
    ("09", "09_model_train_alternative_H4Z5MZEV/alternative_model.pkl", "09_model_train_alternative_H4Z5MZEV/manifest.json"),
]
cands = []
for name, model_key, manifest_key in candidate_keys:
    mp = inputs.get(model_key)
    if mp is None:
        continue
    score = None
    mf = inputs.get(manifest_key)
    if mf is not None:
        try:
            score = find_metric(json.loads(Path(mf).read_text()), primary_metric)
        except Exception:
            score = None
    cands.append((name, score, mp))

latest = inputs.get("latest_model")
lower_better = any(k in primary_metric.lower() for k in ["brier", "log_loss", "rmse", "mae"])
scored = [(s, mp, name) for name, s, mp in cands if s is not None]
if scored:
    if lower_better:
        _, selected_path, sel_name = min(scored, key=lambda x: x[0])
    else:
        _, selected_path, sel_name = max(scored, key=lambda x: x[0])
else:
    selected_path = latest
    sel_name = "latest_model"

with open(selected_path, "rb") as f:
    model = pickle.load(f)

test = pd.read_parquet(inputs["test"])

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = [c for c in test.columns if c not in drop]
    for c in list(feature_cols):
        s = test[c]
        if s.dtype == object and s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)

missing = [c for c in feature_cols if c not in test.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

X_test = test[feature_cols]
y_true = test[target].to_numpy()
proba = model.predict_proba(X_test)
pred = model.predict(X_test)
classes = list(model.classes_)


def ovr_auc(y, p, cls):
    present = set(y)
    vals = []
    for j, c in enumerate(cls):
        if c not in present:
            continue
        yb = (y == c).astype(int)
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, p[:, j]))
    left_out = [c for c in cls if c not in present]
    return (float(np.mean(vals)) if vals else float("nan")), left_out


auc_value, left_out = ovr_auc(y_true, proba, classes)
f1_value = f1_score(y_true, pred, labels=classes, average="macro", zero_division=0)

rng = np.random.default_rng(42)
n = len(test)
boot_auc, boot_f1 = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb, db = y_true[idx], proba[idx], pred[idx]
    a, _ = ovr_auc(yb, pb, classes)
    boot_auc.append(a)
    boot_f1.append(f1_score(yb, db, labels=classes, average="macro", zero_division=0))


def ci(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


auc_lo, auc_hi = ci(boot_auc)
f1_lo, f1_hi = ci(boot_f1)


def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan


metrics_df = pd.DataFrame([
    {"metric": primary_metric, "value": sig4(auc_value), "ci_low": sig4(auc_lo), "ci_high": sig4(auc_hi)},
    {"metric": "f1_macro", "value": sig4(f1_value), "ci_low": sig4(f1_lo), "ci_high": sig4(f1_hi)},
])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Frozen model test metrics with 95% bootstrap CI.")

frame = pd.DataFrame(index=test.index)
frame["row_id"] = test.index
frame["partition"] = "test"
frame["y_true"] = y_true
frame["y_pred"] = pred
for j, c in enumerate(classes):
    frame[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(frame, "test_predictions.parquet")

mlca_runtime.record_metric(primary_metric, auc_value, split="test", ci=(auc_lo, auc_hi))
mlca_runtime.record_metric("f1_macro", f1_value, split="test", ci=(f1_lo, f1_hi))
mlca_runtime.record_fact_candidate("test_roc_auc", auc_value, source="test_metrics.csv")

print(f"selected={sel_name} features={len(feature_cols)} test_rows={len(test)}")
print(f"{primary_metric}={auc_value:.4g} CI=({auc_lo:.4g}, {auc_hi:.4g})")
print(f"f1_macro={f1_value:.4g} CI=({f1_lo:.4g}, {f1_hi:.4g})")
if left_out:
    print(f"test missing classes: {left_out}")
