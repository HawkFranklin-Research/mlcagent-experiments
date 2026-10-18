import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available: {list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
group_cols = [] if group_col is None else [group_col]
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

drop_cols = set(targets) | set(excluded) | set(group_cols)
feature_cols = []
for c in train.columns:
    if c in drop_cols:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        name = str(c).lower()
        if any(k in name for k in ("id", "index", "key", "code")):
            continue
    feature_cols.append(c)

vm = pd.read_csv(inp["09_validate_score_validation_DBTDYA32/validation_metrics.csv"])
cols = {c.lower(): c for c in vm.columns}
if not {"model", "metric", "value"} <= set(cols):
    raise ValueError(f"Unexpected validation metrics columns: {list(vm.columns)}")
mcol, vcol, modcol = cols["metric"], cols["value"], cols["model"]
mask = vm[mcol].astype(str).str.lower() == primary_metric.lower()
if not mask.any():
    mask = vm[mcol].astype(str).str.lower().str.contains(primary_metric.lower(), regex=False)
sub = vm.loc[mask, [modcol, vcol]].copy()
sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
sub = sub.dropna(subset=[vcol]).groupby(modcol, as_index=False)[vcol].mean()
if sub.empty:
    raise ValueError(f"No validation rows for primary metric {primary_metric}")
higher_better = primary_metric.lower() in {"r2", "roc_auc", "auc", "accuracy", "f1", "precision", "recall", "explained_variance"}
best = sub.sort_values(vcol, ascending=not higher_better).iloc[0]
chosen_name = str(best[modcol])
chosen_val = float(best[vcol])

def model_path(name: str):
    s = name.lower()
    if "baseline" in s or "07" in s:
        return inp["07_model_train_baseline_H88BZ8V0/model.pkl"]
    if "alternative" in s or "latest" in s or "08" in s:
        return inp["08_model_train_alternative_BG3XZYK8/alternative_model.pkl"]
    raise ValueError(f"Unknown validation model name: {name}")

with open(model_path(chosen_name), "rb") as f:
    model = pickle.load(f)

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    for attr in ("regressor_", "estimator_", "model_"):
        sub = getattr(obj, attr, None)
        if sub is not None and hasattr(sub, "feature_names_in_"):
            return list(sub.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

feats = feature_names(model)
bad = set(feats or feature_cols) & drop_cols
if bad:
    raise ValueError(f"Target/excluded/group columns in model features: {sorted(bad)}")
if feats is not None:
    missing = [c for c in feats if c not in test.columns]
    if missing:
        raise KeyError(f"Missing test features: {missing}")
    X_test = test[feats].copy()
else:
    X_test = test[feature_cols].copy()

pred = np.asarray(model.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets):
    raise ValueError(f"Prediction shape {pred.shape} does not match targets {targets}")

y_true = test[targets].to_numpy()
per_target = {}
for i, t in enumerate(targets):
    yt, yp = y_true[:, i], pred[:, i]
    per_target[t] = {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
macro = {m: float(np.mean([per_target[t][m] for t in targets])) for m in ("rmse", "mae", "r2")}

rows = []
for t in targets:
    for m in ("rmse", "mae", "r2"):
        rows.append({"model": chosen_name, "metric": f"{m}_{t}", "value": per_target[t][m]})
for m in ("rmse", "mae", "r2"):
    metric_name = primary_metric if m == primary_metric else f"{m}_macro"
    rows.append({"model": chosen_name, "metric": metric_name, "value": macro[m]})
out = pd.DataFrame(rows)
out["value"] = out["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for validation-selected model")

mlca_runtime.record_metric(primary_metric, macro[primary_metric], "test")
mlca_runtime.record_fact_candidate("test_score", macro[primary_metric], source="test_metrics.csv")

print(f"selected {chosen_name} on val {primary_metric}={chosen_val:.4g}")
print(f"test rows={len(test)} features={X_test.shape[1]}")
print(f"test {primary_metric}={macro[primary_metric]:.4g}")
