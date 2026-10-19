import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available task keys: {list(task.keys())}")
primary = str(task["primary_metric"])
group_col = protocol.get("group_column")

val = pd.read_csv(inputs["09_validate_score_validation_8GEY6C8C/validation_metrics.csv"])
def find_col(df, names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    for n in names:
        for l, c in low.items():
            if n in l:
                return c
    return None
model_col = find_col(val, ["model", "model_name", "candidate", "estimator", "name"])
metric_col = find_col(val, ["metric", "metric_name", "measure", "score"])
value_col = find_col(val, ["value", "score_value", "validation_score", "val"])
target_col = find_col(val, ["target", "outcome", "target_name", "label"])
if model_col is None:
    raise KeyError(f"Cannot find model column in validation metrics: {list(val.columns)}")

if metric_col is not None and value_col is not None:
    dfp = val[val[metric_col].astype(str).str.lower().str.contains(primary.lower(), regex=False)].copy()
    if dfp.empty:
        if primary in val.columns:
            scores = {str(k): float(v) for k, v in val.groupby(model_col)[primary].mean().items()}
        else:
            raise KeyError(f"No validation rows for primary metric {primary!r}")
    else:
        macro_names = {"macro", "mean", "average", "avg", "macro_avg", "overall"}
        def group_score(g):
            if target_col is not None:
                ts = g[target_col].astype(str).str.lower()
                mac = ts.isin(macro_names)
                if mac.any():
                    return float(g.loc[mac, value_col].mean())
                return float(g[value_col].mean())
            ms = g[metric_col].astype(str).str.lower()
            exact = ms == primary.lower()
            if exact.any():
                return float(g.loc[exact, value_col].mean())
            mac = ms.str.contains("macro|mean|avg", regex=True)
            if mac.any():
                return float(g.loc[mac, value_col].mean())
            return float(g[value_col].mean())
        scores = {str(name): group_score(g) for name, g in dfp.groupby(model_col)}
elif primary in val.columns:
    scores = {str(k): float(v) for k, v in val.groupby(model_col)[primary].mean().items()}
else:
    raise KeyError(f"Cannot locate primary metric {primary!r} in validation metrics")

if not scores:
    raise ValueError("No validation scores available for model selection")
lower_is_better = {"rmse", "mae", "mse", "mape", "loss", "error", "deviance"}
best_name = min(scores, key=scores.get) if primary.lower() in lower_is_better else max(scores, key=scores.get)

def resolve_model_path(name):
    s = str(name).lower()
    base = inputs["07_model_train_baseline_2S40PTZZ/model.pkl"]
    alt = inputs["08_model_train_alternative_F5KM7TEB/alternative_model.pkl"]
    latest = inputs["latest_model"]
    bm = inputs["07_model_train_baseline_2S40PTZZ/manifest.json"]
    am = inputs["08_model_train_alternative_F5KM7TEB/manifest.json"]
    def has(p):
        try:
            return s in p.read_text().lower()
        except Exception:
            return False
    b, a = has(bm), has(am)
    if b and not a:
        return base
    if a and not b:
        return latest if "latest" in s else alt
    if "base" in s or "07" in s:
        return base
    if "alt" in s or "08" in s:
        return latest if "latest" in s else alt
    if "latest" in s:
        return latest
    if "model.pkl" in s:
        return base
    if "alternative" in s:
        return alt
    raise ValueError(f"Cannot map selected model {name!r} to a saved model path")

model_path = resolve_model_path(best_name)
with open(model_path, "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict) and "model" in model:
    model = model["model"]

test = pd.read_parquet(inputs["test"])
missing_targets = [t for t in targets if t not in test.columns]
if missing_targets:
    raise KeyError(f"Missing target columns in test: {missing_targets}")

def input_columns(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

cols = input_columns(model)
if cols is None:
    cols = []
    for c in test.columns:
        if c in targets or c in excluded or c == group_col:
            continue
        s = test[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
            cl = str(c).lower()
            if any(k in cl for k in ["id", "key", "code", "index"]) or s.is_monotonic_increasing:
                continue
        cols.append(c)
bad = set(cols) & (set(targets) | set(excluded) | ({group_col} if group_col else set()))
if bad:
    raise ValueError(f"Feature columns include targets/excluded/group: {sorted(bad)}")
missing_cols = [c for c in cols if c not in test.columns]
if missing_cols:
    raise KeyError(f"Missing feature columns in test: {missing_cols}")

X = test.loc[:, cols]
pred = np.asarray(model.predict(X))
y_true = test[targets]
per = {}
if len(targets) == 1:
    yt = y_true[targets[0]].to_numpy()
    yp = pred.reshape(-1) if pred.ndim > 1 else pred
    per[targets[0]] = (
        sqrt(mean_squared_error(yt, yp)),
        mean_absolute_error(yt, yp),
        r2_score(yt, yp),
    )
else:
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    for i, t in enumerate(targets):
        yt = y_true[t].to_numpy()
        yp = pred[:, i]
        per[t] = (
            sqrt(mean_squared_error(yt, yp)),
            mean_absolute_error(yt, yp),
            r2_score(yt, yp),
        )
macro = {
    "rmse": float(np.mean([per[t][0] for t in targets])),
    "mae": float(np.mean([per[t][1] for t in targets])),
    "r2": float(np.mean([per[t][2] for t in targets])),
}
if primary not in macro:
    raise ValueError(f"Unsupported primary metric {primary!r}; measured metrics: {sorted(macro)}")

rows = []
for t in targets:
    rmse, mae, r2 = per[t]
    rows.extend([
        (best_name, f"{t}_rmse", rmse),
        (best_name, f"{t}_mae", mae),
        (best_name, f"{t}_r2", r2),
    ])
for m in ["rmse", "mae", "r2"]:
    rows.append((best_name, f"macro_{m}", macro[m]))
rows.append((best_name, primary, macro[primary]))

def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

out = pd.DataFrame(
    [(m, met, sig4(v)) for m, met, v in rows],
    columns=["model", "metric", "value"],
)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for validation-selected model")
primary_value = sig4(macro[primary])
mlca_runtime.record_metric(primary, primary_value, "test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")
print(f"selected model: {best_name}")
print(f"test rows: {len(test)}")
print(f"{primary}: {primary_value:.4f}")
