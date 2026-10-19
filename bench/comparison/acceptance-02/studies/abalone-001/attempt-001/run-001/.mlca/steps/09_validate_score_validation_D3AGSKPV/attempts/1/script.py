import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
available = list(protocol.keys())

try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol.get("group_column")
    primary = task["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {available}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

base_blocked = set(targets) | set(excluded)
features = [c for c in train.columns if c not in base_blocked and (group is None or c != group)]

def is_keylike(s: pd.Series) -> bool:
    if s.isna().any():
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

features = [c for c in features if not is_keylike(train[c])]
X_val = val[features].copy()

def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_MZ4P76F5/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_Q9C0R6MS/selector_all.pkl"])
candidates = {
    "baseline": load_pkl(inp["07_model_train_baseline_X2JY9NC3/model.pkl"]),
    "alternative": load_pkl(inp["08_model_train_alternative_TTKXSSET/alternative_model.pkl"]),
}

def as_pipeline(obj):
    if isinstance(obj, Pipeline) and len(obj.named_steps) > 1:
        return obj
    if isinstance(obj, Pipeline) and len(obj.named_steps) == 1:
        obj = next(iter(obj.named_steps.values()))
    if isinstance(sel, Pipeline) and "preprocessor" in sel.named_steps:
        return Pipeline([("selector", sel), ("model", obj)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", obj)])

def predict_candidate(obj, X):
    pipe = as_pipeline(obj)
    cols = None
    if hasattr(pipe, "feature_names_in_"):
        cols = list(pipe.feature_names_in_)
    elif hasattr(pipe, "named_steps"):
        for step in pipe.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                step_cols = list(step.feature_names_in_)
                if set(step_cols).issubset(set(X.columns)):
                    cols = step_cols
                    break
    return pipe.predict(X[cols] if cols is not None else X)

def score_one(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }

rows = []
primary_by_candidate = {}
for name, obj in candidates.items():
    pred = np.asarray(predict_candidate(obj, X_val))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    by_metric = {"rmse": [], "mae": [], "r2": []}
    for j, target in enumerate(targets):
        mask = val[target].notna().to_numpy()
        scores = score_one(val[target].to_numpy()[mask], pred[mask, j])
        for metric, value in scores.items():
            rows.append({"model": name, "metric": f"{metric}_{target}", "value": value})
            by_metric[metric].append(value)
    for metric in ["rmse", "mae", "r2"]:
        macro = float(np.mean(by_metric[metric]))
        rows.append({"model": name, "metric": f"{metric}_macro", "value": macro})
        if metric == primary:
            primary_by_candidate[name] = macro

lower_better = primary.lower() in {"rmse", "mae", "mse", "mape", "log_loss"}
best_model = (min if lower_better else max)(primary_by_candidate, key=primary_by_candidate.get)
best_score = primary_by_candidate[best_model]

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(lambda x: float(f"{float(x):.4g}"))
mlca_runtime.save_table(
    metrics_df,
    "validation_metrics.csv",
    caption="Validation metrics by candidate and target; macro is the unweighted mean over targets.",
)
mlca_runtime.record_metric(primary, float(best_score), split="val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

print(f"train {train.shape} val {val.shape} features {len(features)}")
print(f"candidates {list(candidates)}")
print(f"best {best_model} {primary}={best_score:.4g}")
