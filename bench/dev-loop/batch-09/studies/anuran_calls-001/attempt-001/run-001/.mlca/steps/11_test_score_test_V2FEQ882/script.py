import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available={list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"]) | set(targets) | {protocol["group_column"]}
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

feature_cols = []
for c in train.columns:
    if c in excluded:
        continue
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)

vm = pd.read_csv(inp["10_validate_score_validation_3J1WH09P/validation_metrics.csv"])
model_col = "model" if "model" in vm.columns else vm.columns[0]
if "metric" in vm.columns:
    sub = vm[vm["metric"].astype(str) == primary]
    if sub.empty:
        sub = vm[vm["metric"].astype(str).str.contains(primary, case=False, regex=False)]
    value_col = "value" if "value" in vm.columns else [c for c in vm.columns if c not in (model_col, "metric")][0]
    vals = sub[value_col].astype(float)
else:
    sub = vm
    vals = vm[primary].astype(float)

higher_better = not any(s in primary.lower() for s in ["loss", "error", "mae", "mse", "rmse"])
idx = vals.idxmax() if higher_better else vals.idxmin()
selected_name = str(sub.loc[idx, model_col])

base_key = "08_model_train_baseline_RDZRN5AQ/model.pkl"
alt_key = "09_model_train_alternative_J8YMJGXB/alternative_model.pkl"
low = selected_name.lower()
if "baseline" in low:
    model_key = base_key
elif "alternative" in low or "alt" in low:
    model_key = alt_key
else:
    model_key = "latest_model"

with open(inp[model_key], "rb") as f:
    complete = pickle.load(f)
if isinstance(complete, dict):
    for k in ("model", "estimator", "pipeline"):
        if k in complete:
            complete = complete[k]
            break

X_test = test.loc[:, feature_cols].copy()
preds = np.asarray(complete.predict(X_test))
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

classes = {t: sorted(pd.unique(train[t].dropna())) for t in targets}
per_f1, per_ham = {}, {}
matches = np.ones(len(test), dtype=bool)
for i, t in enumerate(targets):
    yt = pd.Series(test[t].to_numpy()).reset_index(drop=True)
    yp = pd.Series(preds[:, i]).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(yp) and not pd.api.types.is_numeric_dtype(yt):
        yp = yp.map(lambda v: classes[t][int(v)] if pd.notna(v) and 0 <= int(v) < len(classes[t]) else np.nan)
    per_f1[t] = float(f1_score(yt, yp, average="macro", zero_division=0))
    per_ham[t] = float(np.mean(yt.to_numpy() != yp.to_numpy()))
    matches &= (yt.to_numpy() == yp.to_numpy())

macro_f1 = float(np.mean(list(per_f1.values())))
hamming_avg = float(np.mean(list(per_ham.values())))
subset_acc = float(np.mean(matches))

rows = [
    {"model": selected_name, "metric": primary, "value": macro_f1},
    {"model": selected_name, "metric": "hamming_loss", "value": hamming_avg},
    {"model": selected_name, "metric": "subset_accuracy", "value": subset_acc},
]
out = pd.DataFrame(rows)
out["value"] = out["value"].map(lambda v: float(f"{float(v):.4g}"))
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for validation-selected model")
mlca_runtime.record_metric(primary, macro_f1, split="test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")
print(f"selected={selected_name}")
print(f"n_test={len(test)} features={len(feature_cols)}")
print(f"{primary}={macro_f1:.4f}")
