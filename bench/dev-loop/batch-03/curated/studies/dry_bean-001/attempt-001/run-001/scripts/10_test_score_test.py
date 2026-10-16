import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
task = protocol.get("task", {})
primary_metric = task.get("primary_metric", "roc_auc_ovr")

test_df = pd.read_parquet(inputs["test"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = [c for c in test_df.columns if c not in drop]

X_test = test_df[feature_cols].copy()
y_test = test_df[targets[0]] if targets else None

proba = model.predict_proba(X_test)
pred = model.predict(X_test)
classes = model.classes_

roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
f1 = f1_score(y_test, pred, average="macro", labels=classes)

rng = np.random.default_rng(42)
n = len(y_test)
indices = np.arange(n)
boot_roc = []
boot_f1 = []
for _ in range(1000):
    idx = rng.choice(indices, size=n, replace=True)
    y_b = y_test.iloc[idx]
    proba_b = proba[idx]
    pred_b = pred[idx]
    try:
        r = roc_auc_score(y_b, proba_b, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        r = np.nan
    f = f1_score(y_b, pred_b, average="macro", labels=classes)
    boot_roc.append(r)
    boot_f1.append(f)

ci_roc = (np.nanpercentile(boot_roc, 2.5), np.nanpercentile(boot_roc, 97.5))
ci_f1 = (np.nanpercentile(boot_f1, 2.5), np.nanpercentile(boot_f1, 97.5))

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

metrics_df = pd.DataFrame([
    {"metric": "roc_auc_ovr", "value": roc_auc, "ci_low": ci_roc[0], "ci_high": ci_roc[1]},
    {"metric": "f1_macro", "value": f1, "ci_low": ci_f1[0], "ci_high": ci_f1[1]}
])
for col in ["value", "ci_low", "ci_high"]:
    metrics_df[col] = metrics_df[col].apply(round_sig)

mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary_metric, roc_auc, "test", ci=ci_roc)
mlca_runtime.record_fact_candidate("test_roc_auc", roc_auc, source="test_metrics.csv")

print(f"test rows: {len(test_df)}")
print(f"features: {len(feature_cols)}")
print(f"{primary_metric}: {roc_auc:.4f} CI [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}]")
print(f"f1_macro: {f1:.4f} CI [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
