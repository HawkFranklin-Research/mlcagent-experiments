import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol["task"]
primary = str(task["primary_metric"])
minimize = primary.lower() in {"rmse", "mae", "mse", "mape", "logloss"}

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def find_col(df, names):
    for c in df.columns:
        if str(c).lower() in names:
            return c
    return None

pre = load_pkl(inp["05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_YTKXSJV5/selector_all.pkl"])
baseline = load_pkl(inp["07_model_train_baseline_1RE4HZWD/model.pkl"])
alternative = load_pkl(inp["08_model_train_alternative_22ZP3DFW/alternative_model.pkl"])

val = pd.read_csv(inp["09_validate_score_validation_RP1BFKWM/validation_metrics.csv"])
mc = find_col(val, {"metric", "metric_name"})
modc = find_col(val, {"model", "model_name", "candidate"})
vc = find_col(val, {"value", "score", "metric_value"})
if mc is None or modc is None or vc is None:
    raise ValueError(f"validation_metrics.csv missing metric/model/value columns: {list(val.columns)}")

val["_metric_lower"] = val[mc].astype(str).str.strip().str.lower()
p = primary.lower()
mask = val["_metric_lower"] == p
if not mask.any():
    mask = val["_metric_lower"].str.startswith(p + "_") | val["_metric_lower"].str.startswith(p + " ")
if not mask.any():
    mask = val["_metric_lower"].str.contains(p, regex=False)
if not mask.any():
    raise ValueError(f"no validation rows for primary metric {primary!r}; available={sorted(val['_metric_lower'].unique())[:20]}")

vg = val.loc[mask, [modc, vc]].copy()
vg[vc] = pd.to_numeric(vg[vc], errors="coerce")
vg = vg.dropna(subset=[vc])
vg = vg.groupby(modc, as_index=False)[vc].mean()
best_idx = vg[vc].idxmin() if minimize else vg[vc].idxmax()
best_name = str(vg.loc[best_idx, modc])
best_val = float(vg.loc[best_idx, vc])

candidates = {"baseline": baseline, "alternative": alternative}
def pick_model(name):
    n = str(name).lower()
    if n in candidates:
        return candidates[n]
    if "baseline" in n:
        return candidates["baseline"]
    if "alternative" in n or "latest" in n:
        return candidates["alternative"]
    raise ValueError(f"cannot map validation model {name!r} to {list(candidates)}")

best_model = pick_model(best_name)
test = pd.read_parquet(inp["test"])
raw_cols = input_names(pre) or input_names(sel) or input_names(best_model)
if raw_cols is None:
    raw_cols = [c for c in test.columns if c not in set(targets) | set(excluded) and c != group_col]
    raw_cols = [c for c in raw_cols if not (pd.api.types.is_object_dtype(test[c]) and test[c].nunique(dropna=False) == len(test))]

bad = set(raw_cols) & (set(targets) | set(excluded))
if bad:
    raise ValueError(f"target/excluded columns in feature schema: {sorted(bad)}")
missing = [c for c in raw_cols if c not in test.columns]
if missing:
    raise ValueError(f"test missing required feature columns: {missing}")

def build_pipeline(est, pre, sel, raw_cols):
    raw_set = set(raw_cols)
    if isinstance(est, Pipeline):
        names = input_names(est)
        if names is not None:
            if set(names) == raw_set:
                return est
            return Pipeline([("preprocessor", pre), ("model_pipeline", est)])
        if len(est.named_steps) > 1:
            return est
    steps = []
    sel_names = input_names(sel)
    if sel_names is not None and set(sel_names) == raw_set:
        steps.append(("selector", sel))
        steps.append(("model", est))
    else:
        steps.append(("preprocessor", pre))
        steps.append(("selector", sel))
        steps.append(("model", est))
    return Pipeline(steps)

scorer = build_pipeline(best_model, pre, sel, raw_cols)
X = test[raw_cols]
pred = np.asarray(scorer.predict(X)).reshape(len(test), -1)
if pred.shape[1] < len(targets):
    raise ValueError(f"predictions have {pred.shape[1]} outputs for {len(targets)} targets")

rows = []
rmses, maes, r2s = [], [], []
for i, t in enumerate(targets):
    yt = test[t].to_numpy()
    yp = pred[:, i]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    rmses.append(rmse); maes.append(mae); r2s.append(r2)
    rows += [{"model": best_name, "metric": f"rmse_{t}", "value": rmse},
             {"model": best_name, "metric": f"mae_{t}", "value": mae},
             {"model": best_name, "metric": f"r2_{t}", "value": r2}]

for metric, value in [("rmse", float(np.mean(rmses))),
                      ("mae", float(np.mean(maes))),
                      ("r2", float(np.mean(r2s)))]:
    name = primary if metric.lower() == primary.lower() else f"{metric}_macro"
    rows.append({"model": best_name, "metric": name, "value": value})

df = pd.DataFrame(rows)
df["value"] = df["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics for validation-selected model")

hit = df[df["metric"].astype(str).str.lower() == primary.lower()]
if hit.empty:
    raise ValueError(f"test_metrics.csv has no row for primary metric {primary!r}")
primary_value = float(hit["value"].iloc[0])
mlca_runtime.record_metric(primary, primary_value, split="test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")

print(f"selected={best_name} val_{primary}={best_val:.4g}")
print(f"test_{primary}={primary_value:.4g}")
print(f"rows={len(df)}")
