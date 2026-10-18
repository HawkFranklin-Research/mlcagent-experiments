import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
primary_key = primary.lower()
direction = "minimize" if primary_key in {"rmse", "mae", "mse", "poisson_deviance", "log_loss"} else "maximize"

train = pd.read_parquet(inputs["train"])
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.is_unique and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
        continue
    feature_cols.append(c)

vdf = pd.read_parquet(inputs["09_validate_score_validation_EPZW9V8M/validation_metrics.parquet"])
low = {c.lower(): c for c in vdf.columns}
model_col = next((low[c] for c in ["model", "model_name", "candidate", "estimator", "name"] if c in low), None)
metric_col = next((low[c] for c in ["metric", "metric_name"] if c in low), None)
value_col = next((low[c] for c in ["value", "score", "metric_value"] if c in low), None)
target_col = next((low[c] for c in ["target", "target_name", "outcome"] if c in low), None)
if model_col is None or value_col is None:
    raise KeyError(f"Validation metrics missing model/value columns: {list(vdf.columns)}")
if metric_col:
    mstr = vdf[metric_col].astype(str).str.lower()
    sub = vdf[mstr == primary_key].copy()
    if sub.empty:
        sub = vdf[mstr.str.contains(primary_key, regex=False)].copy()
    if target_col and not sub.empty:
        macro = sub[target_col].astype(str).str.lower().str.contains("macro|mean|avg", regex=True)
        if macro.any():
            sub = sub[macro]
    scores = sub.groupby(model_col)[value_col].mean()
else:
    pcol = low.get(primary_key)
    if pcol is None:
        raise KeyError(f"Primary metric {primary} not in validation metrics: {list(vdf.columns)}")
    scores = vdf.groupby(model_col)[pcol].mean()
if scores.empty:
    raise ValueError("No validation candidate scores found for declared primary metric")
best = scores.idxmin() if direction == "minimize" else scores.idxmax()
best_name = str(best)
bn = best_name.lower()
if "baseline" in bn or "07" in bn or "qgqe" in bn:
    model_path = inputs["07_model_train_baseline_QGQE700X/model.pkl"]
elif "alternative" in bn or "08" in bn or "8p21" in bn:
    model_path = inputs["08_model_train_alternative_8P21XYS1/alternative_model.pkl"]
else:
    raise ValueError(f"Cannot map validation model {best_name} to a saved candidate")

def load_obj(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = load_obj(model_path)
def has_frontend(p):
    return isinstance(p, Pipeline) and any(any(x in k.lower() for x in ("prep", "select", "feature")) for k in p.named_steps)

if has_frontend(model):
    full = model
else:
    prep = load_obj(inputs["05_preprocess_fit_preprocessing_X2VH4MDF/preprocessor.pkl"])
    sel = load_obj(inputs["06_features_retain_features_B66XBBY7/selector_all.pkl"])
    if isinstance(sel, Pipeline) and any("prep" in k.lower() for k in sel.named_steps):
        full = Pipeline(sel.steps + [("model", model)])
    elif isinstance(prep, Pipeline):
        full = Pipeline(prep.steps + [("select", sel), ("model", model)])
    else:
        full = Pipeline([("preprocess", prep), ("select", sel), ("model", model)])

def input_columns(pipe, default):
    if hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    if isinstance(pipe, Pipeline) and pipe.steps:
        first = pipe.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return default

input_cols = input_columns(full, feature_cols)
bad = [c for c in input_cols if c in targets or c in excluded or c == group]
if bad:
    raise ValueError(f"Target/excluded/group column reached model input: {bad}")

test = pd.read_parquet(inputs["test"])
missing = [c for c in input_cols if c not in test.columns]
if missing:
    raise KeyError(f"Missing test feature columns: {missing}")
X_test = test[input_cols]
pred = np.asarray(full.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets):
    raise ValueError(f"Prediction shape {pred.shape} does not match targets {targets}")

rows, metric_vals = [], {}
for i, t in enumerate(targets):
    yt, yp = test[t].to_numpy(), pred[:, i]
    vals = {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
    for name, v in vals.items():
        key = f"{name}_{t}"
        metric_vals[key] = v
        rows.append({"model": best_name, "metric": key, "value": float(f"{v:.4g}")})
for name in ["rmse", "mae", "r2"]:
    macro = float(np.mean([metric_vals[f"{name}_{t}"] for t in targets]))
    metric_vals[f"{name}_macro"] = macro
    rows.append({"model": best_name, "metric": f"{name}_macro", "value": float(f"{macro:.4g}")})

out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics per target and macro average")
for k, v in metric_vals.items():
    mlca_runtime.record_metric(k, v, "test")
headline = metric_vals[f"{primary_key}_macro"]
mlca_runtime.record_metric(primary, headline, "test")
mlca_runtime.record_fact_candidate("test_score", headline, source="test_metrics.csv")
print(f"test rows: {len(test)}; selected: {best_name}; {primary}={headline:.6g}")
