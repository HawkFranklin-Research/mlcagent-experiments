import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Protocol missing {key!r}; available keys: {sorted(protocol)}")
if "primary_metric" not in protocol["task"]:
    raise KeyError(f"task missing primary_metric; task keys: {sorted(protocol['task'])}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

skip = set(targets) | excluded | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns selected from the training table")

missing = [c for c in feature_cols + targets if c not in val.columns]
if missing:
    raise KeyError(f"Validation table missing columns: {missing}")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

preproc = load_pkl(inp["05_preprocess_fit_preprocessing_C8Q2SNQ9/preprocessor.pkl"])
selector = load_pkl(inp["06_features_retain_features_4S5ZAJJA/selector_all.pkl"])
model_paths = {
    "baseline": inp["07_model_train_baseline_VG2WWMW9/model.pkl"],
    "alternative": inp["08_model_train_alternative_3ESX6QWQ/alternative_model.pkl"],
}
models = {name: load_pkl(path) for name, path in model_paths.items()}

def get_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return [str(c) for c in obj.feature_names_in_]
        except Exception:
            pass
    if isinstance(obj, Pipeline) and obj.named_steps:
        return get_cols(list(obj.named_steps.values())[0])
    return None

def same_cols(cols):
    return cols is not None and set(cols) == set(feature_cols)

def predict_candidate(model, val_df):
    mc = get_cols(model)
    if same_cols(mc):
        return model.predict(val_df[mc])
    sc = get_cols(selector)
    if same_cols(sc):
        return model.predict(selector.transform(val_df[sc]))
    pc = get_cols(preproc)
    if same_cols(pc):
        if hasattr(preproc, "set_output"):
            preproc.set_output(transform="pandas")
        Xt = preproc.transform(val_df[pc])
        if selector is not None:
            Xt = selector.transform(Xt)
        return model.predict(Xt)
    return model.predict(val_df[feature_cols])

def score_candidate(model, val_df):
    y_pred = np.asarray(predict_candidate(model, val_df), dtype=float)
    y_true = val_df[targets].to_numpy(dtype=float)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] != len(targets):
        raise ValueError(f"Prediction shape {y_pred.shape} does not match targets {targets}")
    out = {}
    for i, target in enumerate(targets):
        out[f"rmse_{target}"] = mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5
        out[f"mae_{target}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        out[f"r2_{target}"] = r2_score(y_true[:, i], y_pred[:, i])
    for metric in ("rmse", "mae", "r2"):
        out[f"{metric}_macro"] = float(np.mean([out[f"{metric}_{t}"] for t in targets]))
    return out

rows = []
scores = {}
for name, model in models.items():
    scores[name] = score_candidate(model, val)
    for metric, value in scores[name].items():
        rows.append({"model": name, "metric": metric, "value": value})

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].apply(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics per candidate.")

primary_key = f"{primary}_macro"
lower_is_better = primary in {"rmse", "mae", "mse", "mape", "logloss", "neg_mean_squared_error", "neg_root_mean_squared_error"}
best_name = min(scores, key=lambda n: scores[n][primary_key]) if lower_is_better else max(scores, key=lambda n: scores[n][primary_key])
best_score = float(scores[best_name][primary_key])

mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

print(f"train/val rows: {len(train)}, {len(val)}; features: {len(feature_cols)}; targets: {targets}")
print(f"models scored: {list(scores)}")
print(f"best validation {primary}: {best_name} = {best_score:.4g}")
