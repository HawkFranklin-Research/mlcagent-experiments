import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())
targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.is_unique and s.is_monotonic_increasing and s.nunique() == len(s)
    return False

drop = set(targets) | excluded | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
X = val[feature_cols].copy()

with open(inp["05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_W9C3S8F0/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

model_paths = {
    "baseline": inp["07_model_train_baseline_16VA3CWB/model.pkl"],
    "alternative": inp["08_model_train_alternative_YMQ80SAJ/alternative_model.pkl"],
    "latest": inp["latest_model"],
}
models = {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

def first_step(obj):
    if isinstance(obj, Pipeline) and obj.named_steps:
        return obj.named_steps[list(obj.named_steps)[0]]
    return None

def predict_one(m, X):
    for obj in (m, first_step(m)):
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is not None and set(names).issubset(set(X.columns)):
            try:
                return m.predict(X[list(names)])
            except Exception:
                pass
    names = getattr(m, "feature_names_in_", None)
    if names is not None and set(names).issubset(set(X.columns)):
        return m.predict(X[list(names)])
    pre_names = getattr(pre, "feature_names_in_", None)
    Xp = X[list(pre_names)] if pre_names is not None else X
    Xt = pre.transform(Xp)
    if sel is not None:
        sel_names = getattr(sel, "feature_names_in_", None)
        if sel_names is not None and set(sel_names).issubset(set(X.columns)):
            Xt = sel.transform(X[list(sel_names)])
        else:
            Xt = sel.transform(Xt)
    m_names = getattr(m, "feature_names_in_", None)
    if m_names is not None and hasattr(Xt, "columns") and set(m_names).issubset(set(Xt.columns)):
        return m.predict(Xt[list(m_names)])
    return m.predict(Xt)

def score(y_true, y_pred):
    true = np.asarray(y_true)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    pred = np.asarray(y_pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    out = {}
    n = min(true.shape[1], pred.shape[1])
    for i, t in enumerate(targets[:n]):
        out[f"{t}_rmse"] = float(np.sqrt(mean_squared_error(true[:, i], pred[:, i])))
        out[f"{t}_mae"] = float(mean_absolute_error(true[:, i], pred[:, i]))
        out[f"{t}_r2"] = float(r2_score(true[:, i], pred[:, i]))
    for metric in ("rmse", "mae", "r2"):
        vals = [out[f"{t}_{metric}"] for t in targets[:n]]
        out[f"macro_{metric}"] = float(np.mean(vals))
    return out

y_val = val[targets]
all_metrics = {}
rows = []
for name, m in models.items():
    pred = predict_one(m, X)
    met = score(y_val, pred)
    all_metrics[name] = met
    for k, v in met.items():
        rows.append({"model": name, "metric": k, "value": float(f"{v:.4g}")})

mlca_runtime.save_table(
    pd.DataFrame(rows),
    "validation_metrics.csv",
    caption="Validation metrics per candidate and target.",
)

key = f"macro_{primary}"
lower_better = primary in {"rmse", "mae", "mse", "mape"}
if lower_better:
    best_model = min(all_metrics, key=lambda n: all_metrics[n][key])
else:
    best_model = max(all_metrics, key=lambda n: all_metrics[n][key])
best_score = all_metrics[best_model][key]
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

print(f"val rows {len(val)}; candidates {len(models)}; features {len(feature_cols)}")
print(f"best {best_model} {primary}={best_score:.6g}")
