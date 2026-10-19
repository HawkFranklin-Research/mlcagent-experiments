import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

val = pd.read_csv(inp["09_validate_score_validation_JJ0ZPCSX/validation_metrics.csv"])

def choose_candidate(frame, metric):
    low = {c.lower(): c for c in frame.columns}
    model_col = next((low[k] for k in ("model", "candidate", "model_name", "name") if k in low), frame.columns[0])
    if "metric" in low and "value" in low:
        sub = frame[frame[low["metric"]].astype(str).str.lower() == metric.lower()]
        if sub.empty:
            sub = frame[frame[low["metric"]].astype(str).str.contains(metric, case=False, na=False)]
        scores = sub.groupby(model_col)[low["value"]].mean()
    elif metric.lower() in low:
        scores = frame.set_index(model_col)[low[metric.lower()]]
    else:
        cand = [c for c in frame.columns if metric.lower() in c.lower()]
        if not cand:
            raise ValueError(f"validation metrics lack {metric}")
        scores = frame.set_index(model_col)[cand[0]]
    lower_is_better = any(k in metric.lower() for k in ("rmse", "mae", "mse", "mape", "loss", "error"))
    return scores.idxmin() if lower_is_better else scores.idxmax()

best_name = choose_candidate(val, primary)
n = str(best_name).lower()
if "baseline" in n or "07_model" in n or "vg2wwmw9" in n or (n.endswith("model.pkl") and "alternative" not in n):
    model_path = inp["07_model_train_baseline_VG2WWMW9/model.pkl"]
elif "alternative" in n or "08_model" in n or "latest" in n or "3esx6qwq" in n:
    model_path = inp["08_model_train_alternative_3ESX6QWQ/alternative_model.pkl"]
else:
    model_path = inp["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

test = pd.read_parquet(inp["test"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feats = [c for c in test.columns if c not in drop]
safe_feats = []
for c in feats:
    s = test[c]
    nn = int(s.notna().sum())
    if (str(s.dtype) in ("object", "string", "category") or "str" in str(s.dtype)) and nn and s.nunique(dropna=True) == nn:
        continue
    if pd.api.types.is_integer_dtype(s) and nn and s.nunique(dropna=True) == nn and any(k in c.lower() for k in ("id", "key", "index", "code")):
        continue
    safe_feats.append(c)
feats = safe_feats
if hasattr(model, "feature_names_in_"):
    wanted = [c for c in list(model.feature_names_in_) if c in test.columns and c not in drop]
    if wanted:
        feats = wanted

X_test = test[feats].copy()
y_true = test[targets].to_numpy()
pred = np.asarray(model.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1) if len(targets) == 1 else pred.reshape(-1, len(targets))
if pred.shape[1] != len(targets):
    raise ValueError("prediction shape mismatch")

rows = []
macro = {}
for i, target in enumerate(targets):
    yt, yp = y_true[:, i], pred[:, i]
    vals = {
        "rmse": float(mean_squared_error(yt, yp) ** 0.5),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
    for metric, value in vals.items():
        rows.append({"model": str(best_name), "metric": f"{metric}_{target}", "value": float(f"{value:.4g}")})
        macro.setdefault(metric, []).append(value)
for metric, values in macro.items():
    rows.append({"model": str(best_name), "metric": f"{metric}_macro", "value": float(f"{np.mean(values):.4g}")})

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics")
primary_value = float(np.mean(macro[primary])) if primary in macro else float(metrics_df.loc[metrics_df["metric"] == f"{primary}_macro", "value"].iloc[0])
mlca_runtime.record_metric(primary, primary_value, split="test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")
print(f"test_rows={len(test)} features={len(feats)} selected={best_name}")
print(f"{primary}={primary_value:.4g}")
