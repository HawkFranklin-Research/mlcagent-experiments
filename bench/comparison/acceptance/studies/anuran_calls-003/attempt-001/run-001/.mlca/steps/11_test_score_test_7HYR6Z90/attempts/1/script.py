import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

if "task" not in proto:
    raise KeyError(f"Missing protocol key 'task'. Available keys: {list(proto.keys())}")
primary_metric = proto["task"]["primary_metric"]
targets = proto["targets"]
excluded = proto["excluded_columns"]
if "group_column" not in proto:
    raise KeyError(f"Missing protocol key 'group_column'. Available keys: {list(proto.keys())}")
group_col = proto["group_column"]

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 1
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 1
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)

vm = pd.read_csv(inputs["10_validate_score_validation_7QPESJ10/validation_metrics.csv"])
vm_cols = {c.lower(): c for c in vm.columns}
model_col = next(c for k, c in vm_cols.items() if "model" in k or "candidate" in k)
metric_col = next(c for k, c in vm_cols.items() if "metric" in k)
value_col = next(c for k, c in vm_cols.items() if "value" in k or "score" in k)
vm_primary = vm[vm[metric_col].astype(str) == primary_metric].copy()
if vm_primary.empty:
    raise ValueError(f"No validation rows for primary metric {primary_metric!r}")
lower_better = any(x in primary_metric.lower() for x in ("loss", "error", "rmse", "mae"))
best_idx = vm_primary[value_col].idxmin() if lower_better else vm_primary[value_col].idxmax()
selected_name = str(vm_primary.loc[best_idx, model_col])

name_l = selected_name.lower()
if "alt" in name_l or "latest" in name_l or "09_model" in name_l:
    selected_path = inputs["09_model_train_alternative_GFWY19ZA/alternative_model.pkl"]
else:
    selected_path = inputs["08_model_train_baseline_SJFKVA7H/model.pkl"]

with open(selected_path, "rb") as f:
    selected_model = pickle.load(f)
if isinstance(selected_model, dict):
    for k in ("model", "pipeline", "estimator"):
        if k in selected_model:
            selected_model = selected_model[k]
            break

expected = getattr(selected_model, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    missing = [c for c in expected if c not in test.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    X_test = test[expected]
else:
    X_test = test[feature_cols]

y_pred = np.asarray(selected_model.predict(X_test))
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)
if y_pred.shape[1] != len(targets):
    raise ValueError(f"Prediction width {y_pred.shape[1]} does not match targets {len(targets)}")

y_true = test[targets].reset_index(drop=True)
y_pred_df = pd.DataFrame()
for i, t in enumerate(targets):
    p = pd.Series(y_pred[:, i])
    train_vals = set(train[t].dropna().unique())
    pred_vals = set(p.dropna().unique())
    if not pred_vals.issubset(train_vals) and pd.api.types.is_numeric_dtype(p) and not pd.api.types.is_numeric_dtype(train[t]):
        classes = sorted(train[t].dropna().unique())
        p = p.map({j: c for j, c in enumerate(classes)})
    y_pred_df[t] = p
y_pred_df = y_pred_df.reset_index(drop=True)

f1_scores = [f1_score(y_true[t], y_pred_df[t], average="macro", zero_division=0) for t in targets]
macro_f1 = float(np.mean(f1_scores))
hamming_scores = [hamming_loss(y_true[t], y_pred_df[t]) for t in targets]
hamming_mean = float(np.mean(hamming_scores))
subset_acc = float((y_true.astype(str).values == y_pred_df.astype(str).values).all(axis=1).mean())

def r4(x):
    return float(f"{float(x):.4g}")

rows = [
    {"model": selected_name, "metric": primary_metric, "value": r4(macro_f1)},
    {"model": selected_name, "metric": "hamming_loss", "value": r4(hamming_mean)},
    {"model": selected_name, "metric": "subset_accuracy", "value": r4(subset_acc)},
]
for t, s in zip(targets, f1_scores):
    rows.append({"model": selected_name, "metric": f"macro_f1__{t}", "value": r4(s)})
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics for validation-selected model")

mlca_runtime.record_metric(primary_metric, macro_f1, split="test")
mlca_runtime.record_metric("hamming_loss", hamming_mean, split="test")
mlca_runtime.record_metric("subset_accuracy", subset_acc, split="test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")

print(f"selected={selected_name}")
print(f"test_rows={len(test)} {primary_metric}={macro_f1:.4f}")
