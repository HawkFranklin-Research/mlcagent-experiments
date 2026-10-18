import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
target = protocol["targets"][0]
primary = task["primary_metric"]
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

test = pd.read_parquet(inputs["test"])
model = pickle.loads(inputs["latest_model"].read_bytes())

if hasattr(model, "feature_names_in_"):
    feat_cols = list(model.feature_names_in_)
else:
    drop = set(protocol["targets"]) | excluded
    if group:
        drop.add(group)
    feat_cols = []
    for c in test.columns:
        if c in drop:
            continue
        s = test[c]
        if (s.dtype == object or pd.api.types.is_integer_dtype(s)) and s.notna().all() and s.nunique() == len(s):
            continue
        feat_cols.append(c)

X_test = test.loc[:, feat_cols]
y_true = test[target].values

if hasattr(model, "classes_"):
    classes = list(model.classes_)
    pos_label = 1 if 1 in classes else classes[-1]
    pos_idx = classes.index(pos_label)
else:
    pos_label = 1
    pos_idx = 1

prob = model.predict_proba(X_test)[:, pos_idx]
y_bin = (y_true == pos_label).astype(int)

threshold = 0.5
try:
    vm = pd.read_csv(inputs["09_validate_score_validation_06EC0M6T/validation_metrics.csv"])
    if "threshold" in vm.columns:
        vals = vm["threshold"].dropna()
        if len(vals):
            threshold = float(vals.iloc[0])
except Exception:
    pass
try:
    man = json.loads(inputs["09_validate_score_validation_06EC0M6T/manifest.json"].read_text())
    threshold = float(man.get("threshold", man.get("best_threshold", threshold)))
except Exception:
    pass

pred_bin = (prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
values = {
    "roc_auc": roc_auc_score(y_bin, prob),
    "pr_auc": average_precision_score(y_bin, prob),
    "balanced_accuracy": balanced_accuracy_score(y_bin, pred_bin),
    "brier_score": brier_score_loss(y_bin, prob),
    "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
}

metric_names = list(values)
boot = {k: [] for k in metric_names}
rng = np.random.default_rng(42)
n = len(y_bin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb, prb = y_bin[idx], prob[idx], pred_bin[idx]
    if len(np.unique(yb)) < 2:
        continue
    boot["roc_auc"].append(roc_auc_score(yb, pb))
    boot["pr_auc"].append(average_precision_score(yb, pb))
    boot["balanced_accuracy"].append(balanced_accuracy_score(yb, prb))
    boot["brier_score"].append(brier_score_loss(yb, pb))
    tn, fp, fn, tp = confusion_matrix(yb, prb, labels=[0, 1]).ravel()
    boot["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    boot["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

rows = []
for k in metric_names:
    arr = np.array(boot[k], dtype=float)
    arr = arr[~np.isnan(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
    rows.append({"metric": k, "value": values[k], "ci_low": lo, "ci_high": hi})

df = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    df[c] = df[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

for _, r in df.iterrows():
    ci = None if pd.isna(r.ci_low) else [float(r.ci_low), float(r.ci_high)]
    mlca_runtime.record_metric(str(r.metric), float(r.value), "test", ci=ci)

roc_row = df[df.metric == "roc_auc"].iloc[0]
mlca_runtime.record_fact_candidate("test_roc_auc", float(roc_row.value), source="test_metrics.csv")

print(f"test rows: {len(test)}")
print(f"{primary}: {float(roc_row.value):.4g} CI [{float(roc_row.ci_low):.4g}, {float(roc_row.ci_high):.4g}]")
