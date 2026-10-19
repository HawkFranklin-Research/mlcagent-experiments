import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def need(d, key, path):
    if key not in d:
        raise KeyError(f"Missing {path}; available top-level keys: {sorted(d.keys())}")
    return d[key]

protocol = json.loads(inputs["_protocol"].read_text())
targets = need(protocol, "targets", "_protocol['targets']")
excluded = need(protocol, "excluded_columns", "_protocol['excluded_columns']")
task = need(protocol, "task", "_protocol['task']")
primary_metric = need(task, "primary_metric", "_protocol['task']['primary_metric']")

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pkl(inputs["latest_model"])
preprocessor = load_pkl(inputs["05_preprocess_fit_preprocessing_T5EPBF1H/preprocessor.pkl"])
selector = load_pkl(inputs["06_features_retain_features_EWX96D8N/selector_all.pkl"])

def final_est(obj):
    if isinstance(obj, Pipeline):
        steps = list(obj.named_steps.values())
        return steps[-1] if steps else obj
    return obj

def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in names]
    return None

def top_feature(obj, sel):
    est = final_est(obj)
    names = None
    if isinstance(obj, Pipeline):
        try:
            names = list(obj[:-1].get_feature_names_out())
        except Exception:
            names = None
    if names is None and sel is not None:
        try:
            names = list(sel.get_feature_names_out())
        except Exception:
            names = None
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_).ravel()
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_).ravel())
    else:
        return None
    if names is None or len(names) != len(vals):
        raw = getattr(obj, "feature_names_in_", None)
        if raw is not None and len(raw) == len(vals):
            names = [str(x) for x in raw]
        else:
            names = [f"feature_{i}" for i in range(len(vals))]
    return str(names[int(np.argmax(vals))])

est = final_est(model)
features = input_names(model) or input_names(preprocessor) or input_names(selector)
top = top_feature(model, selector)

vm = pd.read_csv(inputs["09_validate_score_validation_PTFC7YN8/validation_metrics.csv"])

def metric_value(df, name):
    if name in df.columns:
        v = pd.to_numeric(df[name], errors="coerce").dropna()
        if len(v):
            return float(v.iloc[0])
    for mcol in ["metric", "metric_name", "name"]:
        if mcol in df.columns:
            mask = df[mcol].astype(str).str.contains(name, case=False, na=False)
            for vcol in ["value", "score", "mean"]:
                if vcol in df.columns:
                    v = pd.to_numeric(df.loc[mask, vcol], errors="coerce").dropna()
                    if len(v):
                        return float(v.iloc[0])
    return None

val = metric_value(vm, primary_metric)
params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
param_items = []
for k in sorted(params)[:8]:
    v = str(params[k])
    if len(v) > 60:
        v = v[:57] + "..."
    param_items.append(f"{k}={v}")
param_text = "; ".join(param_items) if param_items else "not exposed"

def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, (int, float, np.number)):
        try:
            if pd.isna(x):
                return "NA"
        except Exception:
            pass
        return f"{float(x):.4g}"
    return str(x)

features_text = "not exposed"
if features:
    features_text = ", ".join(features[:20]) + ("..." if len(features) > 20 else "")

rows = [
    {"item": "frozen_model_source", "value": "latest_model", "source": "latest_model"},
    {"item": "model_type", "value": type(est).__name__, "source": "latest_model"},
    {"item": "selected_hyperparameters", "value": param_text, "source": "latest_model"},
    {"item": "input_feature_count", "value": len(features) if features else np.nan, "source": "latest_model"},
    {"item": "input_features", "value": features_text, "source": "latest_model"},
    {"item": "validation_metric", "value": primary_metric, "source": "09_validate_score_validation_PTFC7YN8/validation_metrics.csv"},
    {"item": "validation_value", "value": val if val is not None else np.nan, "source": "09_validate_score_validation_PTFC7YN8/validation_metrics.csv"},
    {"item": "validation_rows", "value": len(vm), "source": "09_validate_score_validation_PTFC7YN8/validation_metrics.csv"},
    {"item": "train_rows_used_for_descriptive_reference", "value": len(train_df), "source": "train"},
    {"item": "val_rows_used_for_descriptive_reference", "value": len(val_df), "source": "val"},
    {"item": "limitation", "value": "Observational hourly time series; temporal autocorrelation and weather/calendar confounding limit causal interpretation.", "source": "latest_model"},
    {"item": "limitation", "value": "Interpretation uses the frozen fitted objects only; no new fitting or tuning was performed.", "source": "latest_model"},
    {"item": "limitation", "value": "Validation evidence is from the declared validation split only; feature importance does not imply causality.", "source": "09_validate_score_validation_PTFC7YN8/validation_metrics.csv"},
]
table = pd.DataFrame(rows)
table["value"] = table["value"].map(fmt).astype(str)
table["item"] = table["item"].astype(str)
table["source"] = table["source"].astype(str)

mlca_runtime.save_table(table, "interpretation.csv", caption="Frozen model interpretation: model, hyperparameters, inputs, validation evidence and limitations")
if val is not None:
    mlca_runtime.record_metric(primary_metric, val, "val")
mlca_runtime.record_fact_candidate("top_feature", top if top is not None else "unavailable", source="latest_model")

print(f"train rows: {len(train_df)}, val rows: {len(val_df)}")
print(f"validation {primary_metric}: {val if val is None else f'{val:.4g}'}")
print("saved interpretation.csv")
