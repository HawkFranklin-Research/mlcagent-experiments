import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
primary_metric = protocol["task"]["primary_metric"]

drop = set(targets) | set(excluded)
if protocol.get("group_column"):
    drop.add(protocol["group_column"])

val = pd.read_parquet(inputs["val"])
raw_cols = [c for c in val.columns if c not in drop]

with open(inputs["05_preprocess_fit_preprocessing_CK5QBBYX/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_5J0R0BF8/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

candidates = {
    "baseline": "07_model_train_baseline_V4Y4XVKZ/model.pkl",
    "alternative": "08_model_train_alternative_DHX507BF/alternative_model.pkl",
    "latest_model": "latest_model",
}

def as_pipeline(m):
    if isinstance(m, Pipeline):
        return m
    return Pipeline([("pre", pre), ("sel", sel), ("est", m)])

def predict(pipe, df):
    if hasattr(pipe, "feature_names_in_"):
        cols = list(pipe.feature_names_in_)
        return pipe.predict(df[cols])
    return pipe.predict(df[raw_cols])

def safe_rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

rows = []
preds = {}
for name, path in candidates.items():
    with open(inputs[path], "rb") as f:
        m = pickle.load(f)
    pipe = as_pipeline(m)
    try:
        p = np.asarray(predict(pipe, val))
    except Exception as e:
        print(f"{name}: predict failed ({type(e).__name__})")
        continue
    preds[name] = p
    per = {targets[0]: p} if len(targets) == 1 else {t: p[:, i] for i, t in enumerate(targets)}
    for t in targets:
        yt = val[t].to_numpy()
        yp = per[t]
        prefix = "" if len(targets) == 1 else f"{t}_"
        rows.append({"model": name, "metric": f"{prefix}rmse", "value": safe_rmse(yt, yp)})
        rows.append({"model": name, "metric": f"{prefix}mae", "value": float(mean_absolute_error(yt, yp))})
        rows.append({"model": name, "metric": f"{prefix}r2", "value": float(r2_score(yt, yp))})
    rmses = [safe_rmse(val[t].to_numpy(), per[t]) for t in targets]
    maes = [mean_absolute_error(val[t].to_numpy(), per[t]) for t in targets]
    r2s = [r2_score(val[t].to_numpy(), per[t]) for t in targets]
    rows.append({"model": name, "metric": "macro_rmse", "value": float(np.mean(rmses))})
    rows.append({"model": name, "metric": "macro_mae", "value": float(np.mean(maes))})
    rows.append({"model": name, "metric": "macro_r2", "value": float(np.mean(r2s))})

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics per model")

def get_score(model, metric):
    sub = metrics_df[(metrics_df.model == model) & (metrics_df.metric == metric)]
    return float(sub.value.iloc[0]) if len(sub) else None

headline = primary_metric if len(targets) == 1 else "macro_" + primary_metric
if headline not in set(metrics_df.metric):
    headline = f"{targets[0]}_{primary_metric}"

scores = {m: get_score(m, headline) for m in preds}
scores = {k: v for k, v in scores.items() if v is not None}

lower_better = primary_metric.lower() in ("rmse", "mae", "mse", "log_loss", "mape", "smape", "rmsle")
best_model = (min if lower_better else max)(scores, key=scores.get)
best_score = scores[best_model]

mlca_runtime.record_metric(primary_metric, float(best_score), "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")

print(f"models scored: {list(preds)}")
print(f"rows: {len(metrics_df)}, best: {best_model} ({headline}={best_score:.4f})")
