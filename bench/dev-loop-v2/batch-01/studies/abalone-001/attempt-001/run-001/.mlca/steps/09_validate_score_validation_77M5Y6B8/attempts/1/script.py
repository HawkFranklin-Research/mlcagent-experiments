import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

pp = pickle.load(open(inputs["05_preprocess_fit_preprocessing_EQ66FFY1/preprocessor.pkl"], "rb"))
sel = pickle.load(open(inputs["06_features_retain_features_4S51ZGZN/selector_all.pkl"], "rb"))

def get_input_cols(obj):
    if obj is None: return None
    if hasattr(obj, "feature_names_in_"):
        try: return list(obj.feature_names_in_)
        except Exception: pass
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            c = get_input_cols(step)
            if c is not None: return c
    return None

pp_cols = get_input_cols(pp)
sel_cols = get_input_cols(sel)
feature_cols = pp_cols or sel_cols or [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]
for c in list(feature_cols):
    if c in val.columns:
        s = pd.concat([train[c], val[c]]).dropna()
        if s.nunique() == len(s) and (s.dtype.kind in "iu" or s.dtype.kind == "O"):
            feature_cols.remove(c)

X_val = val[feature_cols].copy()
y_val = {t: val[t].values for t in targets}

def make_predictor(model):
    if model is None: return None
    m_cols = get_input_cols(model)
    if m_cols is not None and len(m_cols) > 0 and set(m_cols).issubset(set(feature_cols)):
        return lambda X: model.predict(X[list(m_cols)])
    if isinstance(model, Pipeline):
        first_cols = get_input_cols(model.steps[0][1])
        if first_cols is not None and set(first_cols).issubset(set(feature_cols)):
            return lambda X: model.predict(X[list(first_cols)])
    steps = []
    sel_cols_here = get_input_cols(sel)
    if sel is not None and sel_cols_here is not None and len(sel_cols_here) > 0 and set(sel_cols_here).issubset(set(feature_cols)):
        steps.append(("sel", sel))
    else:
        if pp is not None: steps.append(("pre", pp))
        if sel is not None: steps.append(("sel", sel))
    steps.append(("model", model))
    full = Pipeline(steps)
    return lambda X: full.predict(X)

def compute_metrics(y_true, y_pred):
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    res = {}
    for i, t in enumerate(targets):
        yt, yp = y_true[t], y_pred[:, i]
        res[t] = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }
    macro = {m: float(np.mean([res[t][m] for t in targets])) for m in ("rmse", "mae", "r2")}
    return res, macro

candidates = {"baseline": pickle.load(open(inputs["07_model_train_baseline_4EB5V15N/model.pkl"], "rb"))}
alt_path = inputs.get("08_model_train_alternative_1TXETTYN/alternative_model.pkl") or inputs.get("latest_model")
if alt_path:
    try: candidates["alternative"] = pickle.load(open(alt_path, "rb"))
    except Exception: pass

rows, scores = [], {}
for name, model in candidates.items():
    try:
        yp = make_predictor(model)(X_val)
        if hasattr(yp, "values"): yp = yp.values
        yp = np.asarray(yp)
        if yp.ndim == 1 and len(targets) > 1: yp = yp.reshape(-1, len(targets))
        res, macro = compute_metrics(y_val, yp)
        scores[name] = macro
        for t in targets:
            for m in ("rmse", "mae", "r2"):
                rows.append({"model": name, "metric": f"{m}_{t}", "value": res[t][m]})
        for m in ("rmse", "mae", "r2"):
            rows.append({"model": name, "metric": f"{m}_macro", "value": macro[m]})
    except Exception as e:
        print(f"skip {name}: {type(e).__name__}")

if not scores: raise RuntimeError("no candidate scored")

key_name = primary_metric.lower() if primary_metric.lower() in ("rmse","mae","r2") else "rmse"
minimize = key_name in ("rmse", "mae", "mse")
best = min(scores, key=lambda n: scores[n][key_name] if minimize else -scores[n][key_name])
best_val = scores[best][key_name]

metrics_df = pd.DataFrame(rows)[["model", "metric", "value"]]
metrics_df["value"] = metrics_df["value"].apply(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics per model and target")

mlca_runtime.record_metric(primary_metric, best_val, "val")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_val, source="validation_metrics.csv")
print(f"candidates={len(scores)} best={best} {primary_metric}={best_val:.4f}")
