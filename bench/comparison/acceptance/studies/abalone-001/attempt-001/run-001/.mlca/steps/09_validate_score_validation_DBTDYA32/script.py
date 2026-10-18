import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(df, c):
    s = df[c]
    if s.nunique(dropna=False) != len(df):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = c.lower()
        return any(t in name for t in ("id", "code", "key", "index"))
    return False

raw_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train, c)
]

with open(inp["05_preprocess_fit_preprocessing_QTKFYNMD/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inp["06_features_retain_features_40Q8191Y/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def get_input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def make_predictor(model):
    if isinstance(model, Pipeline):
        return model
    cols = get_input_cols(model)
    if cols is not None and all(c in raw_cols for c in cols):
        return model
    sel_cols = get_input_cols(selector)
    if sel_cols is not None and all(c in raw_cols for c in sel_cols):
        return Pipeline([("selector", selector), ("model", model)])
    return Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", model)])

candidates = {
    "baseline": inp["07_model_train_baseline_H88BZ8V0/model.pkl"],
    "alternative": inp["08_model_train_alternative_BG3XZYK8/alternative_model.pkl"],
}
models = {}
for name, path in candidates.items():
    with open(path, "rb") as f:
        obj = pickle.load(f)
    models[name] = make_predictor(obj)

val_scored = val.dropna(subset=targets)
y = val_scored[targets].to_numpy(dtype=float)
if len(y) == 0:
    raise ValueError("No validation rows with target values.")

rows = []
macro_scores = {}
for name, model in models.items():
    cols = get_input_cols(model) or raw_cols
    missing = [c for c in cols if c not in val_scored.columns]
    if missing:
        raise ValueError(f"{name} missing validation columns: {missing}")
    pred = np.asarray(model.predict(val_scored[cols]))
    if pred.ndim == 1:
        if len(targets) != 1:
            raise ValueError(f"{name} returned 1D predictions for {len(targets)} targets")
        pred = pred.reshape(-1, 1)
    elif pred.ndim != 2 or pred.shape[1] != len(targets):
        raise ValueError(f"{name} prediction shape {pred.shape} does not match targets")

    metrics = {}
    for j, t in enumerate(targets):
        yt, yp = y[:, j], pred[:, j]
        metrics[f"rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"mae_{t}"] = float(mean_absolute_error(yt, yp))
        metrics[f"r2_{t}"] = float(r2_score(yt, yp))
    for m in ("rmse", "mae", "r2"):
        metrics[f"{m}_macro"] = float(np.mean([metrics[f"{m}_{t}"] for t in targets]))
    macro_scores[name] = {m: metrics[f"{m}_macro"] for m in ("rmse", "mae", "r2")}
    for met, valv in metrics.items():
        rows.append({"model": name, "metric": met, "value": float(f"{valv:.4g}")})

df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics per candidate model.")

if primary not in macro_scores[next(iter(macro_scores))]:
    raise ValueError(f"Primary metric {primary} not computed; available: {list(macro_scores[next(iter(macro_scores))])}")

lower_better = primary in {"rmse", "mae", "mse", "mape", "log_loss"}
higher_better = primary in {"r2", "roc_auc", "accuracy", "f1", "precision", "recall"}
if not (lower_better or higher_better):
    raise ValueError(f"Unknown direction for primary metric {primary}")

best_name = (
    min(macro_scores, key=lambda n: macro_scores[n][primary])
    if lower_better else max(macro_scores, key=lambda n: macro_scores[n][primary])
)
best_value = float(macro_scores[best_name][primary])

mlca_runtime.record_metric(primary, best_value, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_value, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

print(f"val rows scored: {len(val_scored)}")
print(f"candidates: {len(models)}")
print(f"best model: {best_name}")
print(f"{primary}: {best_value:.6f}")
