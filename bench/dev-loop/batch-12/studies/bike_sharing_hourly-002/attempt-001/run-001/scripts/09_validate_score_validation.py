import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

candidates = {
    "baseline": load_pkl(inputs["07_model_train_baseline_GKA0GSZP/model.pkl"]),
    "alternative": load_pkl(inputs["08_model_train_alternative_A4YVEKN6/alternative_model.pkl"]),
    "latest": load_pkl(inputs["latest_model"]),
}

def feature_cols_for(model, frame):
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        return [c for c in fn if c in frame.columns]
    return [c for c in frame.columns if c not in targets and c not in excluded and c != group_col]

higher_better = primary_metric.lower() in ("r2", "roc_auc", "auc", "accuracy", "f1", "precision", "recall")

rows = []
scores = {}
for name, model in candidates.items():
    cols = feature_cols_for(model, val)
    Xv = val[cols]
    pred = model.predict(Xv)
    pred = np.asarray(pred).ravel()
    yv = val[targets].to_numpy()
    per_target = {}
    for i, t in enumerate(targets):
        rmse = float(np.sqrt(mean_squared_error(yv[:, i], pred if len(targets) == 1 else np.asarray(pred)[:, i]))) if len(targets) > 1 else float(np.sqrt(mean_squared_error(yv[:, i], pred)))
        per_target[t] = pred if len(targets) == 1 else np.asarray(pred)[:, i]
    for i, t in enumerate(targets):
        yt = yv[:, i]
        pt = per_target[t]
        rows.append({"model": name, "metric": f"rmse__{t}", "value": round(float(np.sqrt(mean_squared_error(yt, pt))), 4)})
        rows.append({"model": name, "metric": f"mae__{t}", "value": round(float(mean_absolute_error(yt, pt)), 4)})
        rows.append({"model": name, "metric": f"r2__{t}", "value": round(float(r2_score(yt, pt)), 4)})
    # macro averages
    P = np.column_stack([per_target[t] for t in targets])
    if len(targets) > 1:
        rmse_macro = float(np.mean([np.sqrt(mean_squared_error(yv[:, i], P[:, i])) for i in range(len(targets))]))
        mae_macro = float(np.mean([mean_absolute_error(yv[:, i], P[:, i]) for i in range(len(targets))]))
        r2_macro = float(np.mean([r2_score(yv[:, i], P[:, i]) for i in range(len(targets))]))
    else:
        rmse_macro = float(np.sqrt(mean_squared_error(yv[:, 0], P[:, 0])))
        mae_macro = float(mean_absolute_error(yv[:, 0], P[:, 0]))
        r2_macro = float(r2_score(yv[:, 0], P[:, 0]))
    rows.append({"model": name, "metric": "rmse__macro", "value": round(rmse_macro, 4)})
    rows.append({"model": name, "metric": "mae__macro", "value": round(mae_macro, 4)})
    rows.append({"model": name, "metric": "r2__macro", "value": round(r2_macro, 4)})
    # primary metric selection
    key = "rmse__macro" if primary_metric.lower() == "rmse" else f"{primary_metric.lower()}__macro"
    val_map = {r["metric"]: r["value"] for r in rows if r["model"] == name}
    if key in val_map:
        scores[name] = val_map[key]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", "Validation metrics per target and macro")

best_name = max(scores, key=lambda k: scores[k]) if higher_better else min(scores, key=lambda k: scores[k])
best_score = float(scores[best_name])

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"candidates: {list(candidates)}")
print(f"primary_metric={primary_metric} higher_better={higher_better}")
print(f"best={best_name} score={best_score:.4f}")
