import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    task = protocol["task"]
    primary_metric = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing {e} in _protocol; available top-level keys: {list(protocol)}") from e

group_column = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

missing_targets = [t for t in targets if t not in val.columns]
if missing_targets:
    raise KeyError(f"Missing target columns in val: {missing_targets}")

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_column]
for c in list(feature_cols):
    s = train[c]
    is_text_key = (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and s.nunique(dropna=False) == len(train)
    is_int_key = pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train) and any(
        k in c.lower() for k in ("id", "code", "index", "key")
    )
    if is_text_key or is_int_key:
        feature_cols.remove(c)

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_AER96YQP/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_HB34NCSZ/selector_all.pkl")
models = {
    "baseline": load_pkl("07_model_train_baseline_AKKRPP8F/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_HB44V41K/alternative_model.pkl"),
}

def align(X, obj):
    if isinstance(X, pd.DataFrame) and hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        if all(c in X.columns for c in names):
            return X[names]
    return X

def try_predict(fn, n):
    try:
        pred = np.asarray(fn())
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[0] == n and np.isfinite(pred).all():
            return pred
    except Exception:
        pass
    return None

def transform(obj, X):
    if obj is None or not hasattr(obj, "transform"):
        return X
    return obj.transform(align(X, obj) if isinstance(X, pd.DataFrame) else X)

def predict_model(model, frame):
    n = len(frame)
    names = list(getattr(model, "feature_names_in_", []))
    if names and all(c in frame.columns for c in names):
        pred = try_predict(lambda: model.predict(frame[names]), n)
        if pred is not None:
            return pred
    pred = try_predict(lambda: model.predict(frame[feature_cols]), n)
    if pred is not None:
        return pred
    raw = frame[feature_cols] if all(c in frame.columns for c in feature_cols) else frame
    attempts = [
        lambda: model.predict(transform(pre, raw)),
        lambda: model.predict(transform(sel, raw)),
        lambda: model.predict(transform(sel, transform(pre, raw))),
    ]
    for fn in attempts:
        pred = try_predict(fn, n)
        if pred is not None:
            return pred
    raise ValueError("Could not produce validation predictions")

def score_model(name, pred):
    y_true = val[targets].to_numpy()
    y_pred = np.asarray(pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    rows = []
    for j, target in enumerate(targets):
        values = {
            "rmse": np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j])),
            "mae": mean_absolute_error(y_true[:, j], y_pred[:, j]),
            "r2": r2_score(y_true[:, j], y_pred[:, j]),
        }
        for metric, value in values.items():
            rows.append({"model": name, "metric": f"{metric}_{target}", "value": float(value)})
    for metric in ("rmse", "mae", "r2"):
        vals = [r["value"] for r in rows if r["metric"].startswith(metric + "_")]
        rows.append({"model": name, "metric": f"{metric}_macro", "value": float(np.mean(vals))})
    return rows

rows = []
for name, model in models.items():
    rows.extend(score_model(name, predict_model(model, val)))

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].apply(lambda v: float(f"{v:.4g}") if np.isfinite(v) else np.nan)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", "Validation metrics per model and target")

macro_metric = f"{primary_metric}_macro"
candidates = metrics_df[metrics_df["metric"] == macro_metric].copy()
higher_is_better = primary_metric.lower() in {
    "r2", "r2_score", "roc_auc", "accuracy", "f1", "precision", "recall",
    "balanced_accuracy", "average_precision",
}
best_idx = candidates["value"].idxmax() if higher_is_better else candidates["value"].idxmin()
best = candidates.loc[best_idx]
best_model = str(best["model"])
best_score = float(best["value"])

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"Scored {len(models)} models on val; rows={len(metrics_df)}")
print(f"Best val {primary_metric}: {best_model} {best_score:.4g}")
