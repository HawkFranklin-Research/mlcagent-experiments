import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
primary = proto["task"]["primary_metric"]
lower_better = primary.lower() in {"rmse", "mae", "mse", "mape"}

val = pd.read_csv(inp["09_validate_score_validation_8TJRCB1S/validation_metrics.csv"])
mcol = next(c for c in val.columns if c.lower() == "metric")
modcol = next(c for c in val.columns if c.lower() == "model")
vcol = next(c for c in val.columns if c.lower() == "value")
exact = val[val[mcol].astype(str).str.lower() == primary.lower()]
sub = exact if len(exact) else val[val[mcol].astype(str).str.lower().str.contains(primary.lower())]
scores = sub.groupby(modcol)[vcol].mean()
best_name = scores.idxmin() if lower_better else scores.idxmax()

with open(inp["07_model_train_baseline_25GK6Z71/model.pkl"], "rb") as f:
    baseline = pickle.load(f)
with open(inp["08_model_train_alternative_VF3B36TN/alternative_model.pkl"], "rb") as f:
    alternative = pickle.load(f)

name = str(best_name).lower()
if "baseline" in name or "07" in name:
    model, label = baseline, "baseline"
elif "alternative" in name or "08" in name:
    model, label = alternative, "alternative"
else:
    raise ValueError(f"Cannot map validation model {best_name}")

test = pd.read_parquet(inp["test"])
excluded = set(proto.get("excluded_columns", []))
group = proto.get("group_column")
feature_cols = [c for c in test.columns if c not in targets and c not in excluded and c != group]

expected = getattr(model, "feature_names_in_", None)
if expected is None and hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            expected = step.feature_names_in_
            break
if expected is not None:
    feature_cols = [c for c in expected if c in test.columns]

pred = model.predict(test[feature_cols])
pred = pred.values if isinstance(pred, pd.DataFrame) else np.asarray(pred)
if pred.ndim == 1 and len(targets) == 1:
    pred = pred.reshape(-1, 1)

rows = []
for i, t in enumerate(targets):
    y = test[t].to_numpy()
    p = pred[:, i] if pred.ndim > 1 else pred
    rows.append({"model": label, "metric": f"rmse_{t}", "value": float(np.sqrt(mean_squared_error(y, p)))})
    rows.append({"model": label, "metric": f"mae_{t}", "value": float(mean_absolute_error(y, p))})
    rows.append({"model": label, "metric": f"r2_{t}", "value": float(r2_score(y, p))})

for metric in ["rmse", "mae", "r2"]:
    vals = [r["value"] for r in rows if r["metric"].startswith(metric + "_")]
    if vals:
        rows.append({"model": label, "metric": metric, "value": float(np.mean(vals))})

out = pd.DataFrame(rows)
out["value"] = out["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics")

test_score = float(out.loc[out["metric"] == primary, "value"].iloc[0])
mlca_runtime.record_metric(primary, test_score, split="test")
mlca_runtime.record_fact_candidate("test_score", test_score, source="test_metrics.csv")
print(f"selected={label} primary={primary} test_score={test_score:.6g} rows={len(out)}")
