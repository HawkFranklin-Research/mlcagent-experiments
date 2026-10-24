import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
task = protocol["task"]
primary = task["primary_metric"]

def read_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

def is_key_like(s):
    if s.isna().any() or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        v = s.astype(int)
        return v.is_monotonic_increasing and (v.max() - v.min() + 1 == len(v))
    return False

feature_cols = [c for c in train.columns
                if c not in set(targets) | set(excluded)
                and c != group and not is_key_like(train[c])]

pre = read_pkl("05_preprocess_fit_preprocessing_3WJD77DF/preprocessor.pkl")
sel = read_pkl("06_features_retain_features_6D2RD7T4/selector_all.pkl")
model = read_pkl("latest_model")

def is_raw(obj):
    names = list(getattr(obj, "feature_names_in_", []))
    return len(names) > 0 and set(names) == set(feature_cols)

if isinstance(model, Pipeline) and (is_raw(model) or not hasattr(model, "feature_names_in_")):
    final = model
else:
    steps = []
    if is_raw(sel):
        steps.append(("selector", sel))
    elif is_raw(pre):
        steps.append(("preprocessor", pre))
        if sel is not None:
            steps.append(("selector", sel))
    else:
        if pre is not None:
            steps.append(("preprocessor", pre))
        if sel is not None:
            steps.append(("selector", sel))
    steps.append(("model", model))
    final = Pipeline(steps)

est = final.named_steps[list(final.named_steps)[-1]] if isinstance(final, Pipeline) else final
classes = list(est.classes_)
if len(classes) != 2:
    raise ValueError("Binary task expected two model classes")
pos_idx = 0
pos_label = task.get("positive_label", None)
if pos_label is not None and pos_label in classes:
    pos_idx = classes.index(pos_label)
else:
    pos_idx = 1
    pos_label = classes[pos_idx]
neg_label = classes[1 - pos_idx]

names = list(getattr(final, "feature_names_in_", []))
X_test = test[names] if names else test[feature_cols]
y_true = test[targets[0]].to_numpy()
proba = final.predict_proba(X_test)
proba_pos = proba[:, pos_idx]
y_bin = (y_true == pos_label).astype(int)

def threshold_from_validation():
    vm = pd.read_parquet(inputs["09_validate_score_validation_J88R8Z29/validation_metrics.parquet"])
    for c in vm.columns:
        if "threshold" in c.lower():
            v = pd.to_numeric(vm[c], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[0])
    for c in vm.columns:
        if vm[c].dtype == object:
            mask = vm[c].astype(str).str.contains("threshold", case=False, na=False)
            if mask.any():
                for v in vm[mask].iloc[0].values:
                    try:
                        fv = float(v)
                        if 0.0 <= fv <= 1.0:
                            return fv
                    except Exception:
                        pass
    val = pd.read_parquet(inputs["val"])
    Xv = val[names] if names else val[feature_cols]
    pv = final.predict_proba(Xv)[:, pos_idx]
    yv = (val[targets[0]].to_numpy() == pos_label).astype(int)
    fpr, tpr, thrs = roc_curve(yv, pv)
    return float(thrs[int(np.argmax(tpr - fpr))])

threshold = threshold_from_validation()
y_pred = np.where(proba_pos >= threshold, pos_label, neg_label)

def metric_values(y, yp, yb, pp):
    out = {}
    if len(np.unique(yb)) == 2:
        out["roc_auc"] = roc_auc_score(yb, pp)
        out["pr_auc"] = average_precision_score(yb, pp)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    out["brier_score"] = brier_score_loss(yb, pp)
    out["balanced_accuracy"] = balanced_accuracy_score(y, yp) if len(np.unique(y)) == 2 else np.nan
    tp = np.sum((y == pos_label) & (yp == pos_label))
    fn = np.sum((y == pos_label) & (yp != pos_label))
    tn = np.sum((y != pos_label) & (yp != pos_label))
    fp = np.sum((y != pos_label) & (yp == pos_label))
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out

metric_names = [primary] + [m for m in
    ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
    if m != primary]
full = metric_values(y_true, y_pred, y_bin, proba_pos)
rng = np.random.default_rng(42)
boot = {m: [] for m in metric_names}
n = len(y_true)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    vals = metric_values(y_true[idx], y_pred[idx], y_bin[idx], proba_pos[idx])
    for m in metric_names:
        v = vals.get(m, np.nan)
        if np.isfinite(v):
            boot[m].append(v)

rows = []
for m in metric_names:
    vals = boot[m]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    rows.append({"metric": m, "value": full.get(m, np.nan), "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

primary_row = metrics_df.loc[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(primary_row["value"]), "test",
                           ci=(float(primary_row["ci_low"]), float(primary_row["ci_high"])))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(primary_row["value"]), source="test_metrics.csv")

pred_df = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": y_true,
    "y_pred": y_pred,
})
for j, cls in enumerate(classes):
    pred_df[f"proba__{cls}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")
print("test rows", len(test), "features", len(feature_cols))
print("primary", primary, round(float(primary_row["value"]), 4))
