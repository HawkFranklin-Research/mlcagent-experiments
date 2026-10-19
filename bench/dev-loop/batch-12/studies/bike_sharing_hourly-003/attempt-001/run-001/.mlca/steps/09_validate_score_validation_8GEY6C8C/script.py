import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def load_pkl(key):
    with open(paths[key], "rb") as f:
        return pickle.load(f)

protocol = json.loads(paths["_protocol"].read_text())
missing = [k for k in ("targets", "excluded_columns", "task") if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def is_key_like(s):
    if s.isna().any():
        return False
    n = len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique() == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

pre = load_pkl("05_preprocess_fit_preprocessing_D4ZRP9PK/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_Z3FWD4WP/selector_all.pkl")
candidates = {
    "baseline": load_pkl("07_model_train_baseline_2S40PTZZ/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_F5KM7TEB/alternative_model.pkl"),
}

def frontend_transform(X):
    if sel is not None:
        sn = list(getattr(sel, "feature_names_in_", []))
        if sn and all(c in X.columns for c in sn):
            return sel.transform(X[sn])
    if pre is not None:
        pn = list(getattr(pre, "feature_names_in_", []))
        cols = pn if pn and all(c in X.columns for c in pn) else feature_cols
        Z = pre.transform(X[cols])
    else:
        Z = X[feature_cols]
    if sel is not None:
        Z = sel.transform(Z)
    return Z

def predict_candidate(model, X):
    names = list(getattr(model, "feature_names_in_", []))
    if names and all(c in X.columns for c in names):
        return model.predict(X[names])
    if isinstance(model, Pipeline):
        try:
            return model.predict(X[feature_cols])
        except Exception:
            pass
    return model.predict(frontend_transform(X))

def score_model(model):
    p = np.asarray(predict_candidate(model, val))
    y = val[targets].to_numpy(dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    rows = []
    per_metric = {"rmse": [], "mae": [], "r2": []}
    for i, t in enumerate(targets):
        yt, yp = y[:, i], p[:, i]
        vals = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }
        for m, v in vals.items():
            rows.append((m, t, v))
            per_metric[m].append(v)
    for m, vals in per_metric.items():
        rows.append((m, "macro", float(np.mean(vals))))
    return rows

all_rows = []
macro_scores = {}
for name, model in candidates.items():
    rows = score_model(model)
    for metric, target_name, value in rows:
        all_rows.append({"model": name, "metric": f"{metric}_{target_name}", "value": value})
    macro_scores[name] = {m: v for m, t, v in rows if t == "macro"}

lower_better = primary.lower() in {"rmse", "mae", "mse", "log_loss", "rmsle", "mape"}
if lower_better:
    best_model = min(macro_scores, key=lambda k: macro_scores[k][primary])
else:
    best_model = max(macro_scores, key=lambda k: macro_scores[k][primary])
best_score = float(macro_scores[best_model][primary])

metrics_df = pd.DataFrame(all_rows, columns=["model", "metric", "value"])
metrics_df["value"] = metrics_df["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", "Validation metrics by model and target")

mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

print(f"train rows={len(train)}, val rows={len(val)}, targets={targets}")
print(f"feature_cols={len(feature_cols)}")
print(f"selected={best_model} val_{primary}={best_score:.4g}")
