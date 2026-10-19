import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from math import isfinite
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from pandas.api.types import is_object_dtype, is_string_dtype, is_integer_dtype

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
top = list(proto.keys())
for k in ["targets", "excluded_columns", "task"]:
    if k not in proto:
        raise KeyError(f"Missing _protocol[{k!r}]; available top-level keys: {top}")
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing _protocol['task']['primary_metric']; task keys: {list(task.keys())}")
primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])
for t in targets:
    if t not in test.columns:
        raise KeyError(f"Target {t!r} missing from test")

def feature_columns(df, targets, excluded, group):
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    n = len(df)
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if (is_object_dtype(s) or is_string_dtype(s)) and s.nunique(dropna=True) == n:
            continue
        if is_integer_dtype(s) and s.nunique(dropna=True) == n:
            if s.min() == 0 and s.max() == n - 1 and s.is_unique:
                continue
        cols.append(c)
    return cols

feature_cols = feature_columns(train, targets, excluded, group)

vdf = pd.read_csv(inputs["09_validate_score_validation_D3AGSKPV/validation_metrics.csv"])
def find_col(df, names):
    for n in names:
        for c in df.columns:
            if str(c).lower() == n.lower():
                return c
    for n in names:
        for c in df.columns:
            if n.lower() in str(c).lower():
                return c
    return None
model_col = find_col(vdf, ["model", "candidate", "model_name"])
metric_col = find_col(vdf, ["metric", "metric_name"])
value_col = find_col(vdf, ["value", "score"])
if not all([model_col, metric_col, value_col]):
    raise KeyError(f"Validation metrics columns not found; columns={list(vdf.columns)}")
v = vdf[vdf[metric_col].astype(str).str.lower() == primary.lower()].copy()
if v.empty:
    v = vdf[vdf[metric_col].astype(str).str.lower().str.contains(primary.lower(), na=False)].copy()
if v.empty:
    raise ValueError(f"Primary metric {primary!r} not found in validation metrics")
v[value_col] = pd.to_numeric(v[value_col], errors="coerce")
v = v.dropna(subset=[value_col])
if v.empty:
    raise ValueError("No numeric validation scores for primary metric")
lower_better = primary.lower() in {"rmse", "mae", "mse", "mape", "log_loss", "mean_squared_error", "mean_absolute_error"}
agg = v.groupby(model_col, dropna=False)[value_col].mean().reset_index()
agg = agg.sort_values(value_col, ascending=lower_better)
best_name = str(agg.iloc[0][model_col])
best_val = float(agg.iloc[0][value_col])

def model_path_for(name, inputs):
    nl = name.lower()
    if "baseline" in nl:
        return inputs["07_model_train_baseline_X2JY9NC3/model.pkl"]
    if "alternative" in nl or "latest" in nl:
        return inputs["08_model_train_alternative_TTKXSSET/alternative_model.pkl"]
    for k, p in inputs.items():
        kl = k.lower()
        if (nl in kl or kl in nl) and str(p).endswith(".pkl") and "model" in kl:
            return p
    return inputs["latest_model"]

model_path = model_path_for(best_name, inputs)
with open(model_path, "rb") as f:
    model = pickle.load(f)
if not (hasattr(model, "named_steps") and len(model.named_steps) > 0):
    pre = pickle.load(open(inputs["05_preprocess_fit_preprocessing_MZ4P76F5/preprocessor.pkl"], "rb"))
    sel = pickle.load(open(inputs["06_features_retain_features_Q9C0R6MS/selector_all.pkl"], "rb"))
    pre_in = getattr(pre, "feature_names_in_", None)
    sel_in = getattr(sel, "feature_names_in_", None)
    raw = set(map(str, feature_cols))
    if sel_in is not None and all(str(c) in raw for c in sel_in):
        model = Pipeline([("selector", sel), ("model", model)])
    elif pre_in is not None and all(str(c) in raw for c in pre_in):
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])
    else:
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

feat_names = getattr(model, "feature_names_in_", None)
if feat_names is not None:
    feat_names = [str(c) for c in feat_names]
    missing = [c for c in feat_names if c not in test.columns]
    if missing:
        raise KeyError(f"Model expects missing test columns: {missing}")
    X_test = test.loc[:, feat_names].copy()
else:
    X_test = test.loc[:, feature_cols].copy()

y_pred = model.predict(X_test)
metric_funcs = {
    "rmse": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
    "mae": lambda yt, yp: float(mean_absolute_error(yt, yp)),
    "r2": lambda yt, yp: float(r2_score(yt, yp)),
}
rows, per_target = [], {m: [] for m in metric_funcs}
for i, t in enumerate(targets):
    yt = test[t].to_numpy()
    if y_pred.ndim == 1:
        yp = y_pred
    else:
        yp = y_pred[:, i] if y_pred.shape[1] > 1 else y_pred.ravel()
    for m, fn in metric_funcs.items():
        val = fn(yt, yp)
        per_target[m].append(val)
        rows.append({"model": best_name, "metric": f"{m}[{t}]", "value": val})
macro_vals = {}
for m, vals in per_target.items():
    mv = float(np.mean(vals))
    macro_vals[m] = mv
    metric_name = primary if m.lower() == primary.lower() else m
    rows.append({"model": best_name, "metric": metric_name, "value": mv})
if primary.lower() not in macro_vals:
    raise ValueError(f"Primary metric {primary!r} not among computed metrics {list(macro_vals)}")
primary_value = float(macro_vals[primary.lower()])
def sig4(x):
    return float(f"{x:.4g}") if isfinite(x) else x
for r in rows:
    r["value"] = sig4(r["value"])
out = pd.DataFrame(rows)[["model", "metric", "value"]]
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics")
primary_sig = sig4(primary_value)
mlca_runtime.record_metric(primary, primary_sig, "test")
mlca_runtime.record_fact_candidate("test_score", primary_sig, source="test_metrics.csv")
print(f"selected={best_name} val_{primary}={best_val:.6g}")
print(f"test_rows={len(test)} targets={targets}")
print(f"test_{primary}={primary_sig:.6g}")
