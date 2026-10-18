import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime as rt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

rt.set_seed(42)
inp = rt.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
primary = protocol["task"]["primary_metric"]
direction = "min" if primary in ("rmse","mae","mse","mape","logloss") else "max"

val = pd.read_csv(inp["09_validate_score_validation_FBE2D3SM/validation_metrics.csv"])
mlow = val["metric"].astype(str).str.lower()
pm = val[(mlow == primary.lower()) | mlow.str.match(rf"^{primary.lower()}_")]
if len(pm) == 0:
    pm = val[mlow.str.startswith(primary.lower())]
per_model = pm.groupby("model")["value"].mean()
best = per_model.idxmin() if direction == "min" else per_model.idxmax()
print("candidates:", list(per_model.index), "best:", best)

model_path = inp["07_model_train_baseline_RHMAVZ3Z/model.pkl"] if "base" in str(best).lower() else inp["08_model_train_alternative_PZKEGXY4/alternative_model.pkl"]
with open(model_path, "rb") as f:
    model = pickle.load(f)
if not isinstance(model, Pipeline):
    with open(inp["05_preprocess_fit_preprocessing_Y0BA94GV/preprocessor.pkl"], "rb") as f: pre = pickle.load(f)
    with open(inp["06_features_retain_features_J9XJJ7CH/selector_all.pkl"], "rb") as f: sel = pickle.load(f)
    model = Pipeline([("pre", pre), ("sel", sel), ("model", model)])

test = pd.read_parquet(inp["test"])
excluded = protocol.get("excluded_columns", []) or []
group_col = protocol.get("group_column")
feat_cols = [c for c in test.columns if c not in targets and c not in excluded and c != group_col]
feat_cols = [c for c in feat_cols if not (pd.api.types.is_integer_dtype(test[c]) and test[c].nunique() == len(test) and len(test) > 1)]
fn = getattr(model, "feature_names_in_", None)
if fn is not None:
    feat_cols = [c for c in fn if c in test.columns]
X_test = test[feat_cols]
print("X_test:", X_test.shape)

y_pred = model.predict(X_test)
y_true = test[targets].to_numpy()
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

def sig4(x): return float(f"{x:.4g}")

rows, primary_scores = [], []
for i, t in enumerate(targets):
    yt, yp = y_true[:, i], y_pred[:, i]
    r = float(np.sqrt(mean_squared_error(yt, yp)))
    m = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    for nm, v in [("rmse", r), ("mae", m), ("r2", r2)]:
        rows.append({"model": best, "metric": f"{nm}_{t}", "value": sig4(v)})
    primary_scores.append({"rmse": r, "mae": m, "r2": r2}[primary])

macro = {
    "rmse": float(np.mean([np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(len(targets))])),
    "mae":  float(np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(len(targets))])),
    "r2":   float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(len(targets))])),
}
for mn, mv in macro.items():
    rows.append({"model": best, "metric": f"{mn}_macro", "value": sig4(mv)})

rt.save_table(pd.DataFrame(rows), "test_metrics.csv", "Test metrics per target and macro average")
score = float(np.mean(primary_scores))
rt.record_metric(primary, score, "test")
rt.record_fact_candidate("test_score", score, source="test_metrics.csv")
print("test", primary, "macro =", round(score, 4))
