import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column", None)
primary = proto["task"]["primary_metric"]

test = pd.read_parquet(inp["test"])

vmet = pd.read_csv(inp["09_validate_score_validation_WEK9FDVR/validation_metrics.csv"])
vmet.columns = [c.lower() for c in vmet.columns]
for alt, new in [("model_name", "model"), ("candidate", "model"), ("name", "model"),
                 ("metric_name", "metric"), ("measure", "metric"),
                 ("metric_value", "value"), ("score", "value"), ("val", "value")]:
    if new not in vmet.columns and alt in vmet.columns:
        vmet = vmet.rename(columns={alt: new})

def best_model(vmet, primary):
    if "model" not in vmet.columns:
        raise ValueError(f"No model column. Columns: {list(vmet.columns)}")
    for m in [primary, primary + "_macro", primary + "_mean"]:
        if "metric" in vmet.columns and "value" in vmet.columns:
            mask = vmet["metric"].astype(str).str.lower() == m.lower()
            if mask.any():
                sub = vmet[mask]
                agg = sub.groupby("model")["value"].mean().reset_index()
                return str(agg.loc[agg["value"].idxmin(), "model"])
    if primary in vmet.columns:
        return str(vmet.loc[vmet[primary].idxmin(), "model"])
    if "metric" in vmet.columns and "value" in vmet.columns:
        mask = vmet["metric"].astype(str).str.lower().str.contains(primary.lower(), na=False)
        if mask.any():
            sub = vmet[mask]
            agg = sub.groupby("model")["value"].mean().reset_index()
            return str(agg.loc[agg["value"].idxmin(), "model"])
    cand = [c for c in vmet.columns if primary.lower() in c.lower() and c != "model"]
    if cand:
        return str(vmet.loc[vmet[cand[0]].idxmin(), "model"])
    raise ValueError(f"Cannot find {primary} in validation metrics. Columns: {list(vmet.columns)}")

best_name = best_model(vmet, primary)

cands = {
    "baseline": inp["07_model_train_baseline_F3SAKS84/model.pkl"],
    "alternative": inp["08_model_train_alternative_G7VTXE7A/alternative_model.pkl"],
}
if "baseline" in best_name.lower():
    key = "baseline"
elif "alternative" in best_name.lower():
    key = "alternative"
else:
    raise ValueError(f"Unknown model name: {best_name}")
with open(cands[key], "rb") as f:
    model = pickle.load(f)

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in test.columns if c not in drop]
if hasattr(model, "feature_names_in_"):
    feat_cols = [c for c in model.feature_names_in_ if c in test.columns and c not in drop]
elif hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            feat_cols = [c for c in step.feature_names_in_ if c in test.columns and c not in drop]
            break

X_test = test[feat_cols]
y_true = test[targets].to_numpy()
y_pred = np.asarray(model.predict(X_test))
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

rows = []
for i, t in enumerate(targets):
    yt, yp = y_true[:, i], y_pred[:, i]
    rows.append({"model": best_name, "metric": f"rmse_{t}", "value": float(np.sqrt(mean_squared_error(yt, yp)))})
    rows.append({"model": best_name, "metric": f"mae_{t}", "value": float(mean_absolute_error(yt, yp))})
    rows.append({"model": best_name, "metric": f"r2_{t}", "value": float(r2_score(yt, yp))})

for m in ["rmse", "mae", "r2"]:
    vals = [r["value"] for r in rows if r["metric"].startswith(m + "_")]
    rows.append({"model": best_name, "metric": f"{m}_macro", "value": float(np.mean(vals))})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics per target and macro average")
macro_rmse = float(df.loc[df["metric"] == "rmse_macro", "value"].iloc[0])
mlca_runtime.record_metric("rmse", macro_rmse, "test")
mlca_runtime.record_fact_candidate("test_score", macro_rmse, source="test_metrics.csv")
print(f"test rows: {len(test)}, features: {len(feat_cols)}")
print(f"selected: {best_name}, test rmse macro: {macro_rmse:.4f}")
