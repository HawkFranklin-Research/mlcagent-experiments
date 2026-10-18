import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing _protocol['{key}']; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
primary_metric = protocol["task"].get("primary_metric")
if not primary_metric:
    raise ValueError("Missing _protocol['task']['primary_metric']")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
for t in targets:
    if t not in val.columns:
        raise KeyError(f"Target '{t}' missing from val")

blocked = set(targets) | excluded | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in blocked]
for c in list(feature_cols):
    s = train[c]
    all_distinct = s.notna().sum() > 1 and s.nunique(dropna=True) == s.notna().sum()
    if all_distinct and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        feature_cols.remove(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_2K81ED4E/selector_all.pkl"])

def make_predictor(obj, raw_cols):
    if isinstance(obj, Pipeline):
        return obj
    if hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        if set(raw_cols).issubset(set(names)):
            return obj
    if hasattr(obj, "n_features_in_") and obj.n_features_in_ == len(raw_cols):
        return obj
    if pre is not None and sel is not None:
        return Pipeline([("preprocessor", pre), ("selector", sel), ("model", obj)])
    return obj

def align_X(predictor, X):
    names = None
    if hasattr(predictor, "feature_names_in_"):
        names = list(predictor.feature_names_in_)
    elif isinstance(predictor, Pipeline) and predictor.steps:
        first = predictor.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            names = list(first.feature_names_in_)
    if names:
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        return X[names]
    return X

def metrics_for(y_true, y_pred, target_names):
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    out = {}
    for i, t in enumerate(target_names):
        yt = y_true[t].to_numpy()
        yp = y_pred[:, i]
        out[t] = {
            "rmse": root_mean_squared_error(yt, yp),
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp),
        }
    for m in ["rmse", "mae", "r2"]:
        out.setdefault("macro", {})[m] = float(np.mean([out[t][m] for t in target_names]))
    return out

candidates = {
    "baseline": inputs["07_model_train_baseline_GKP580AC/model.pkl"],
    "alternative": inputs["08_model_train_alternative_4WYQPWAG/alternative_model.pkl"],
}
X_val = val[feature_cols].copy()
y_val = val[targets].copy()
rows = []
scores = {}

for name, path in candidates.items():
    obj = load_pkl(path)
    predictor = make_predictor(obj, feature_cols)
    pred = predictor.predict(align_X(predictor, X_val))
    m = metrics_for(y_val, pred, targets)
    scores[name] = m
    for target_key, vals in m.items():
        for metric_name, value in vals.items():
            label = f"{metric_name}_{target_key}"
            rows.append({"model": name, "metric": label, "value": float(f"{value:.4g}")})

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by model")

key = primary_metric.lower()
if key not in {"rmse", "mae", "r2"}:
    raise ValueError(f"Unsupported primary_metric '{primary_metric}'")
lower_better = key != "r2"
best_name = (min if lower_better else max)(scores, key=lambda n: scores[n]["macro"][key])
best_score = float(scores[best_name]["macro"][key])

mlca_runtime.record_metric(primary_metric, best_score, split="val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, unit=primary_metric, source="validation_metrics.csv")
print(f"scored {len(candidates)} models on val; {len(metrics_df)} metric rows")
print(f"best {primary_metric}={best_score:.4g} ({best_name})")
