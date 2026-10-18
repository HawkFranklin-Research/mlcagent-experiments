import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task key 'primary_metric'; available task keys: {list(task)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]
primary = task["primary_metric"]
direction = {"rmse": "min", "mae": "min", "r2": "max", "roc_auc": "max"}.get(primary)
if direction is None:
    raise ValueError(f"Unsupported primary metric direction for {primary!r}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        return len(s) > 1 and s.nunique(dropna=True) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in set(targets) and c not in excluded and c != group_col and not is_keylike(train[c])
]
X_val = val[feature_cols].copy()
y_val = val[targets].copy()

with open(inp["05_preprocess_fit_preprocessing_9HEDMWQR/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inp["06_features_retain_features_HD824TRJ/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inp["07_model_train_baseline_WTZ11RHD/model.pkl"], "rb") as f:
    baseline_model = pickle.load(f)
with open(inp["08_model_train_alternative_KHZYBTMD/alternative_model.pkl"], "rb") as f:
    alternative_model = pickle.load(f)

def make_predictor(model):
    if isinstance(model, Pipeline):
        return model
    if isinstance(selector, Pipeline):
        return Pipeline([("selector", selector), ("model", model)])
    return Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", model)])

def expected_columns(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if isinstance(est, Pipeline):
        for step in est.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def sig4(x):
    return float(f"{x:.4g}")

candidates = [("baseline", baseline_model), ("alternative", alternative_model)]
rows = []
primary_scores = {}

for name, model in candidates:
    est = make_predictor(model)
    cols = expected_columns(est)
    Xc = X_val[cols] if cols is not None else X_val
    pred = est.predict(Xc)
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    per_target = {}
    for i, target in enumerate(targets):
        yt = y_val[target].to_numpy()
        pt = pred[:, i]
        vals = {
            "rmse": np.sqrt(mean_squared_error(yt, pt)),
            "mae": mean_absolute_error(yt, pt),
            "r2": r2_score(yt, pt),
        }
        per_target[target] = vals
        for metric, value in vals.items():
            rows.append({"model": name, "metric": f"{metric}_{target}", "value": sig4(value)})
    macro = {}
    for metric in ["rmse", "mae", "r2"]:
        macro[metric] = float(np.mean([per_target[t][metric] for t in targets]))
        rows.append({"model": name, "metric": f"{metric}_macro", "value": sig4(macro[metric])})
    primary_scores[name] = macro[primary]

best_name = min(primary_scores, key=primary_scores.get) if direction == "min" else max(primary_scores, key=primary_scores.get)
best_score = primary_scores[best_name]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", "Validation metrics by candidate and target.")
mlca_runtime.record_metric(primary, float(best_score), split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")

print(f"rows={len(metrics_df)} candidates={len(candidates)}")
print(f"best_model={best_name}")
print(f"{primary}_val={best_score:.4f}")
