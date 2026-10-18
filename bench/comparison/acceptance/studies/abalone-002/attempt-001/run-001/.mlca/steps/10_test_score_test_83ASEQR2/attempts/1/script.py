import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
task = protocol["task"]
primary_metric = (task.get("primary_metric") or "rmse")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def key_like(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s):
        return False
    return True

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

val = pd.read_csv(inputs["09_validate_score_validation_5RG1R1QC/validation_metrics.csv"])
def find_col(df, names):
    for c in df.columns:
        if c.lower() in names:
            return c
    return None

model_col = find_col(val, {"model", "model_name", "candidate", "candidate_name", "pipeline"})
metric_col = find_col(val, {"metric", "metric_name", "name"})
value_col = find_col(val, {"value", "score", "metric_value"})
if not all([model_col, metric_col, value_col]):
    raise ValueError(f"Validation metrics columns unavailable: {list(val.columns)}")
tgt_col = find_col(val, {"target", "outcome", "target_name"})
v = val[val[metric_col].astype(str).str.lower() == primary_metric.lower()].copy()
if v.empty:
    v = val[val[metric_col].astype(str).str.lower().str.contains(primary_metric.lower(), regex=False)].copy()
if v.empty:
    raise ValueError(f"No rows for primary metric {primary_metric!r}")
if tgt_col:
    tv = v[tgt_col].astype(str).str.lower()
    if (tv == "macro").any():
        v = v[tv == "macro"]
v[value_col] = pd.to_numeric(v[value_col], errors="coerce")
v = v.dropna(subset=[value_col])
maximize = primary_metric.lower() in {"r2", "roc_auc", "accuracy", "f1", "precision", "recall", "average_precision"}
scores = v.groupby(model_col)[value_col].mean()
best_name = scores.sort_values(ascending=not maximize).index[0]
low = str(best_name).lower()
if "baseline" in low:
    model_path, model_label = inputs["07_model_train_baseline_GKP580AC/model.pkl"], "baseline"
elif "alternative" in low or "latest" in low:
    model_path, model_label = inputs["08_model_train_alternative_4WYQPWAG/alternative_model.pkl"], "alternative"
else:
    raise ValueError(f"Unknown validation model {best_name!r}; available: {list(scores.index)}")

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = load_pickle(model_path)
pre = load_pickle(inputs["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"])
sel = load_pickle(inputs["06_features_retain_features_2K81ED4E/selector_all.pkl"])

def first_fni(obj):
    fni = getattr(obj, "feature_names_in_", None)
    if fni is None and isinstance(obj, Pipeline) and obj.steps:
        fni = getattr(obj.steps[0][1], "feature_names_in_", None)
    return None if fni is None else [str(c) for c in fni]

def accepts_raw(obj):
    fni = first_fni(obj)
    return fni is not None and set(fni).issubset(set(feature_cols))

def front_steps():
    if isinstance(sel, Pipeline):
        keys = [k.lower() for k in sel.named_steps]
        if any("preprocess" in k or "preproc" in k for k in keys):
            return [("features", sel)]
    return [("preprocessor", pre), ("selector", sel)]

if isinstance(model, Pipeline) and accepts_raw(model):
    predictor = model
elif isinstance(model, Pipeline):
    keys = [k.lower() for k in model.named_steps]
    has_pre = any("preprocess" in k or "preproc" in k for k in keys)
    has_sel = any("select" in k or "feature" in k for k in keys)
    if has_pre:
        predictor = model
    elif has_sel:
        predictor = Pipeline([("preprocessor", pre), *model.steps])
    else:
        predictor = Pipeline([*front_steps(), *model.steps])
else:
    predictor = Pipeline([*front_steps(), ("model", model)])

test = pd.read_parquet(inputs["test"])
X = test[feature_cols].copy()
exp = first_fni(predictor)
if exp is not None:
    missing = [c for c in exp if c not in X.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    X = X[exp]
mask = test[targets].notna().all(axis=1)
X = X.loc[mask]
pred = np.asarray(predictor.predict(X))
if len(targets) == 1:
    pred = pred.reshape(-1)
    preds = [pred]
else:
    if pred.ndim == 1:
        pred = pred.reshape(-1, len(targets))
    preds = [pred[:, i] for i in range(len(targets))]

rows = []
per = {m: [] for m in ("rmse", "mae", "r2")}
for target, yhat in zip(targets, preds):
    y = test.loc[mask, target].to_numpy(dtype=float)
    vals = {
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }
    for m, vv in vals.items():
        per[m].append(vv)
        rows.append({"model": model_label, "metric": f"{target}_{m}", "value": float(f"{vv:.4g}")})
macro = {m: float(np.nanmean(per[m])) for m in per}
for m, vv in macro.items():
    rows.append({"model": model_label, "metric": f"macro_{m}", "value": float(f"{vv:.4g}")})
out = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics")
primary_value = macro.get(primary_metric.lower(), macro["rmse"])
mlca_runtime.record_metric(primary_metric, float(primary_value), "test")
mlca_runtime.record_fact_candidate("test_score", float(primary_value), source="test_metrics.csv")
print(f"selected={model_label} n_test={len(X)}")
print(f"{primary_metric}={primary_value:.4g}")
