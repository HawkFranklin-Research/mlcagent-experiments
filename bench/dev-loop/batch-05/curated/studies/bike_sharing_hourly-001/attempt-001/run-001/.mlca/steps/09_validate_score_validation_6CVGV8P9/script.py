import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]
lower_is_better = primary in {"rmse", "mae", "mse", "log_loss", "mean_squared_error"}

val = pd.read_parquet(inp["val"])

cands = {
    "baseline_07": inp["07_model_train_baseline_MY9YJS0R/model.pkl"],
    "alternative_08": inp["08_model_train_alternative_VYCC8X2X/alternative_model.pkl"],
}

def raw_cols(model, df):
    cols = None
    if hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
        except Exception:
            cols = None
    if cols is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                cols = list(step.feature_names_in_)
                break
    if cols is None:
        drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
        cols = [c for c in df.columns if c not in drop]
    return [c for c in cols if c in df.columns]

y = val[targets[0]].to_numpy()

rows = []
summary = {}
for name, path in cands.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    X = val[raw_cols(model, val)]
    pred = np.asarray(model.predict(X)).ravel()
    m = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }
    summary[name] = m
    for k, v in m.items():
        rows.append({"model": name, "metric": f"{k}_macro", "value": round(v, 6)})
        rows.append({"model": name, "metric": f"{k}_{targets[0]}", "value": round(v, 6)})

selector = min if lower_is_better else max
best_name = selector(summary, key=lambda n: summary[n][primary])
best_score = float(summary[best_name][primary])

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation metrics per candidate (macro and per-target)")

for name, m in summary.items():
    for k, v in m.items():
        mlca_runtime.record_metric(f"{k}_{name}", float(v), "val")

mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"val_rows={len(val)} candidates={len(cands)}")
print(f"best={best_name} {primary}={best_score:.4f}")
