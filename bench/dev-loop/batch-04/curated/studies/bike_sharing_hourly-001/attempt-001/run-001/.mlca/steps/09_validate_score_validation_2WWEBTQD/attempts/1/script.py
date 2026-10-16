import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
primary_metric = task["primary_metric"]
group_col = protocol.get("group_column")

val = pd.read_parquet(inputs["val"])
train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

all_cols = list(train.columns)
key_like = [c for c in all_cols if c not in targets and c not in excluded and c != group_col and is_key_like(train[c])]
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col and c not in key_like]

preproc = pickle.loads(inputs["05_preprocess_fit_preprocessing_DVNTE848/preprocessor.pkl"].read_bytes())
selector = pickle.loads(inputs["06_features_retain_features_TBVVBD2H/selector_all.pkl"].read_bytes())

def get_input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def build_predictor(model):
    if isinstance(model, Pipeline):
        input_cols = get_input_cols(model)
        if input_cols is not None and set(input_cols).issubset(set(train.columns)):
            return model
    steps = []
    if preproc is not None:
        steps.append(("preproc", preproc))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)

def compute_metric(name, y_true, y_pred):
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    if name == "poisson_deviance":
        yp = np.clip(y_pred, 1e-10, None)
        return float(mean_poisson_deviance(y_true, yp))
    if name == "mean_squared_error":
        return float(mean_squared_error(y_true, y_pred))
    if name == "mean_absolute_error":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {name}")

candidates = {
    "baseline": inputs["07_model_train_baseline_5MMHKFNM/model.pkl"],
    "alternative": inputs["08_model_train_alternative_SHDNC9V9/alternative_model.pkl"],
}

metric_names = ["rmse", "mae", "r2"]
if primary_metric not in metric_names:
    metric_names.append(primary_metric)

rows = []
scores = {}
for name, path in candidates.items():
    model = pickle.loads(path.read_bytes())
    predictor = build_predictor(model)
    input_cols = get_input_cols(predictor) or feature_cols
    missing = [c for c in input_cols if c not in val.columns]
    if missing:
        raise KeyError(f"Missing columns for {name}: {missing}")
    X = val[input_cols].copy()
    y_pred = predictor.predict(X)
    if len(targets) == 1:
        y_true = val[targets[0]].to_numpy()
        y_pred = np.asarray(y_pred).ravel()
    else:
        y_true = val[targets].to_numpy()
        y_pred = np.asarray(y_pred)

    metrics = {}
    for i, t in enumerate(targets):
        yt, yp = (y_true, y_pred) if len(targets) == 1 else (y_true[:, i], y_pred[:, i])
        metrics[t] = {m: compute_metric(m, yt, yp) for m in metric_names}
    macro = {m: float(np.mean([metrics[t][m] for t in targets])) for m in metric_names}
    for t in targets:
        for m, v in metrics[t].items():
            rows.append({"model": name, "metric": f"{m}_{t}", "value": v})
    for m, v in macro.items():
        rows.append({"model": name, "metric": f"{m}_macro", "value": v})
    rows.append({"model": name, "metric": primary_metric, "value": macro[primary_metric]})
    scores[name] = macro[primary_metric]

direction = task.get("primary_metric_direction", task.get("metric_direction"))
if direction is None:
    direction = "minimize" if primary_metric in ["rmse", "mae", "poisson_deviance", "mean_squared_error", "mean_absolute_error"] else "maximize"

best_model = min(scores, key=scores.get) if direction == "minimize" else max(scores, key=scores.get)
best_score = scores[best_model]

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics", caption="Validation metrics per model and target")
mlca_runtime.record_metric(primary_metric, best_score, split="val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

print(f"val shape: {val.shape}")
print(f"best model: {best_model}")
print(f"{primary_metric}: {best_score:.4f}")
