import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from sklearn.metrics import poisson_deviance
except ImportError:
    from sklearn.metrics import mean_poisson_deviance as poisson_deviance

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = protocol.get("excluded_columns", []) or []
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]
p_lower = primary.lower()
lower_better = {"poisson_deviance": True, "rmse": True, "mae": True, "r2": False}
if p_lower not in lower_better:
    raise ValueError(f"Unknown primary metric: {primary}")

val = pd.read_csv(inp["09_validate_score_validation_08S6B52Y/validation_metrics.csv"])
cand = val[val["metric"].astype(str).str.lower().str.contains(p_lower, regex=False)].copy()
if cand.empty:
    raise ValueError(f"No validation rows for primary metric {primary}")
cand["value"] = pd.to_numeric(cand["value"], errors="coerce")
agg = cand.groupby("model", as_index=False)["value"].mean().sort_values("value", ascending=lower_better[p_lower])
chosen = str(agg.iloc[0]["model"])
n = chosen.lower()
if "baseline" in n:
    model_path = inp["07_model_train_baseline_4E4949CT/model.pkl"]
elif "alternative" in n:
    model_path = inp["08_model_train_alternative_4C73HMKJ/alternative_model.pkl"]
elif "latest" in n:
    model_path = inp["latest_model"]
else:
    raise ValueError(f"Cannot map model {chosen!r} to a declared path")

def read_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = read_pkl(model_path)
pre = read_pkl(inp["05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl"])
sel = read_pkl(inp["06_features_retain_features_R3TMNHH4/selector_all.pkl"])

def complete(m):
    return isinstance(m, Pipeline) and any(("preprocess" in s.lower() or "selector" in s.lower()) for s in m.named_steps)

if complete(model):
    predictor = model
elif isinstance(sel, Pipeline) and any("preprocess" in s.lower() for s in sel.named_steps):
    predictor = Pipeline([("frontend", sel), ("model", model)])
else:
    predictor = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

test = pd.read_parquet(inp["test"])
drop = [c for c in list(targets) + list(excluded) + ([group] if group else []) if c in test.columns]
X_test = test.drop(columns=drop)
if hasattr(predictor, "feature_names_in_"):
    X_test = X_test.loc[:, list(predictor.feature_names_in_)]
pred = np.asarray(predictor.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets):
    if len(targets) == 1 and pred.shape[1] >= 1:
        pred = pred[:, [0]]
    else:
        raise ValueError(f"Prediction shape {pred.shape} does not match targets {targets}")

def sig4(x):
    return float(f"{float(x):.4g}")

rows, rmse_v, mae_v, r2_v, prim_v = [], [], [], [], []
for j, t in enumerate(targets):
    y = test[t].to_numpy()
    yp = pred[:, j]
    rmse = mean_squared_error(y, yp) ** 0.5
    mae = mean_absolute_error(y, yp)
    r2 = r2_score(y, yp)
    if p_lower == "poisson_deviance":
        pv = poisson_deviance(y, np.clip(yp, 1e-9, None))
    elif p_lower == "rmse":
        pv = rmse
    elif p_lower == "mae":
        pv = mae
    else:
        pv = r2
    rmse_v.append(rmse); mae_v.append(mae); r2_v.append(r2); prim_v.append(pv)
    rows += [
        {"model": chosen, "metric": f"RMSE_{t}", "value": sig4(rmse)},
        {"model": chosen, "metric": f"MAE_{t}", "value": sig4(mae)},
        {"model": chosen, "metric": f"R2_{t}", "value": sig4(r2)},
        {"model": chosen, "metric": f"{primary}_{t}", "value": sig4(pv)},
    ]
pm = float(np.mean(prim_v))
rows += [
    {"model": chosen, "metric": "RMSE_macro", "value": sig4(np.mean(rmse_v))},
    {"model": chosen, "metric": "MAE_macro", "value": sig4(np.mean(mae_v))},
    {"model": chosen, "metric": "R2_macro", "value": sig4(np.mean(r2_v))},
    {"model": chosen, "metric": primary, "value": sig4(pm)},
]
out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary, pm, "test")
mlca_runtime.record_fact_candidate("test_score", pm, source="test_metrics.csv")
print(f"selected_model={chosen}")
print(f"test_rows={len(test)} features={X_test.shape[1]} targets={len(targets)}")
print(f"{primary}={pm:.6g}")
