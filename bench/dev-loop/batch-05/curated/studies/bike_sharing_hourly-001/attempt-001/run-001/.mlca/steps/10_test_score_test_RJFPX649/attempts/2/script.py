import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

test = pd.read_parquet(inputs["test"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_6CVGV8P9/validation_metrics.csv"])

lower_better = primary.lower() in ("rmse", "mae", "mse", "log_loss", "mape")

best_model = None
if primary in val_metrics.columns:
    best_model = str(val_metrics.sort_values(primary, ascending=lower_better).iloc[0]["model"])
elif "metric" in val_metrics.columns:
    mcol = val_metrics["metric"].astype(str).str.lower()
    pm = val_metrics[mcol == primary.lower()]
    if pm.empty:
        pm = val_metrics[mcol.str.startswith(primary.lower())]
    if not pm.empty:
        best_model = str(pm.sort_values("value", ascending=lower_better).iloc[0]["model"])
if best_model is None:
    best_model = "latest_model"

if "baseline" in best_model.lower():
    model_path = inputs["07_model_train_baseline_MY9YJS0R/model.pkl"]
elif "alternative" in best_model.lower():
    model_path = inputs["08_model_train_alternative_VYCC8X2X/alternative_model.pkl"]
else:
    model_path = inputs["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

feature_cols = None
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
elif hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            feature_cols = list(step.feature_names_in_)
            break
if feature_cols is None:
    feature_cols = [c for c in test.columns
                    if c not in targets and c not in excluded and c != group_col]

X_test = test[feature_cols]
preds = np.asarray(model.predict(X_test))
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

rows = []
per_target = {}
for i, t in enumerate(targets):
    yt = test[t].to_numpy()
    pt = preds[:, i]
    rmse = float(np.sqrt(mean_squared_error(yt, pt)))
    mae = float(mean_absolute_error(yt, pt))
    r2 = float(r2_score(yt, pt))
    per_target[t] = {"rmse": rmse, "mae": mae, "r2": r2}
    for m, v in [("rmse", rmse), ("mae", mae), ("r2", r2)]:
        rows.append({"model": best_model, "metric": f"{m}_{t}", "value": v})

for m in ["rmse", "mae", "r2"]:
    macro = float(np.mean([per_target[t][m] for t in targets]))
    rows.append({"model": best_model, "metric": f"{m}_macro", "value": macro})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "test_metrics.csv", "Test metrics per target and macro average")

test_score = float(np.mean([per_target[t][primary] for t in targets]))
mlca_runtime.record_metric(primary, test_score, "test")
mlca_runtime.record_fact_candidate("test_score", test_score, source="test_metrics.csv")

print(f"test rows: {len(test)}")
print(f"selected model: {best_model}")
print(f"{primary} test: {test_score:.4f}")
