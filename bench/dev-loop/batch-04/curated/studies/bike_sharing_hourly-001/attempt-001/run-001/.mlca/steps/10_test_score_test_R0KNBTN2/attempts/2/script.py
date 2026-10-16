import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)
primary_metric = protocol["task"]["primary_metric"]

test = pd.read_parquet(inputs["test"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_2WWEBTQD/validation_metrics.csv"])

lower_is_better = primary_metric in ["poisson_deviance", "rmse", "mae", "mean_squared_error", "log_loss"]
if "direction" in protocol["task"]:
    lower_is_better = protocol["task"]["direction"] == "lower"
elif "primary_metric_direction" in protocol["task"]:
    lower_is_better = protocol["task"]["primary_metric_direction"] == "lower"

val_primary = val_metrics[val_metrics["metric"] == primary_metric]
if val_primary.empty:
    raise ValueError(f"Primary metric {primary_metric} not found in validation metrics")
val_primary = val_primary.sort_values("value", ascending=lower_is_better)
best_model_name = val_primary.iloc[0]["model"]

model_map = {
    "baseline": inputs["07_model_train_baseline_5MMHKFNM/model.pkl"],
    "alternative": inputs["08_model_train_alternative_SHDNC9V9/alternative_model.pkl"],
    "latest": inputs["latest_model"],
}
model_path = None
for key, path in model_map.items():
    if key in best_model_name.lower():
        model_path = path
        break
if model_path is None:
    if best_model_name in model_map:
        model_path = model_map[best_model_name]
    else:
        model_path = inputs["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

feature_cols = [c for c in test.columns if c not in targets and c not in excluded and c != group_col]
X_test = test[feature_cols]

try:
    preds = model.predict(X_test)
except Exception:
    with open(inputs["05_preprocess_fit_preprocessing_DVNTE848/preprocessor.pkl"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(inputs["06_features_retain_features_TBVVBD2H/selector_all.pkl"], "rb") as f:
        selector = pickle.load(f)
    X_test_pre = preprocessor.transform(X_test)
    X_test_sel = selector.transform(X_test_pre)
    preds = model.predict(X_test_sel)

if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

def compute_metric(name, y_true, y_pred):
    if name == "poisson_deviance":
        return mean_poisson_deviance(y_true, y_pred)
    elif name == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif name == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif name == "r2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {name}")

metrics = []
for i, target in enumerate(targets):
    y_true = test[target].values
    y_pred = preds[:, i] if preds.ndim > 1 else preds
    for m in ["RMSE", "MAE", "R2"]:
        val = compute_metric(m.lower(), y_true, y_pred)
        metrics.append({"model": best_model_name, "metric": f"{m}_{target}", "value": val})

for m in ["RMSE", "MAE", "R2"]:
    vals = [x["value"] for x in metrics if x["metric"].startswith(m + "_")]
    if vals:
        metrics.append({"model": best_model_name, "metric": f"{m}_macro", "value": np.mean(vals)})

primary_vals = []
for i, target in enumerate(targets):
    y_true = test[target].values
    y_pred = preds[:, i] if preds.ndim > 1 else preds
    primary_vals.append(compute_metric(primary_metric, y_true, y_pred))
primary_val = np.mean(primary_vals)
metrics.append({"model": best_model_name, "metric": primary_metric, "value": primary_val})

metrics_df = pd.DataFrame(metrics)
metrics_df["value"] = metrics_df["value"].apply(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary_metric, primary_val, split="test")
mlca_runtime.record_fact_candidate("test_score", primary_val, source="test_metrics.csv")

print(f"Test shape: {test.shape}")
print(f"Selected model: {best_model_name}")
print(f"{primary_metric}: {primary_val:.4g}")
