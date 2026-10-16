import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def pget(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}. Available top-level keys: {list(d.keys())}")
    return d[key]

targets = pget(protocol, "targets")
excluded = pget(protocol, "excluded_columns")
group = protocol.get("group_column")
task = pget(protocol, "task")
primary_metric = pget(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ["model", "estimator", "pipeline"]:
            if k in obj:
                return obj[k]
    return obj

latest = unwrap(load_pkl("latest_model"))
baseline = unwrap(load_pkl("07_model_train_baseline_5MMHKFNM/model.pkl"))
alt = unwrap(load_pkl("08_model_train_alternative_SHDNC9V9/alternative_model.pkl"))
pre = load_pkl("05_preprocess_fit_preprocessing_DVNTE848/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_TBVVBD2H/selector_all.pkl")
val_metrics = pd.read_csv(inputs["09_validate_score_validation_2WWEBTQD/validation_metrics.csv"])

def final_est(obj):
    if hasattr(obj, "named_steps"):
        return list(obj.named_steps.values())[-1]
    return obj

def get_params(obj):
    est = final_est(obj)
    return est.get_params() if hasattr(est, "get_params") else {}

def get_top_feature(obj):
    est = final_est(obj)
    names = None
    if hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
    elif hasattr(obj, "get_feature_names_out"):
        try:
            names = list(obj.get_feature_names_out())
        except Exception:
            pass
    if names is None:
        for o in [pre, sel]:
            if hasattr(o, "get_feature_names_out"):
                try:
                    names = list(o.get_feature_names_out())
                    break
                except Exception:
                    pass
    imp = None
    if hasattr(est, "coef_"):
        coef = np.abs(est.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        imp = coef
    elif hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    if imp is not None and names is not None and len(names) == len(imp):
        return str(names[int(np.argmax(imp))])
    return "unknown"

top_feature = get_top_feature(latest)
rows = []

def add(section, item, value, source):
    rows.append({"section": str(section), "item": str(item), "value": str(value), "source": str(source)})

add("model", "frozen_model_type", type(latest).__name__, "latest_model")
add("model", "baseline_model_type", type(baseline).__name__, "07_model_train_baseline_5MMHKFNM/model.pkl")
add("model", "alternative_model_type", type(alt).__name__, "08_model_train_alternative_SHDNC9V9/alternative_model.pkl")

params = get_params(latest)
for k in ["n_estimators", "max_depth", "learning_rate", "alpha", "l1_ratio", "C", "solver", "loss", "max_iter", "min_samples_leaf", "subsample", "colsample_bytree"]:
    if k in params:
        add("hyperparameters", k, params[k], "latest_model")

add("inputs", "targets", ", ".join(targets), "_protocol")
add("inputs", "excluded_columns", ", ".join(excluded), "_protocol")
add("inputs", "group_column", group, "_protocol")
add("inputs", "primary_metric", primary_metric, "_protocol")
add("inputs", "train_rows", len(train), "train")
add("inputs", "val_rows", len(val), "val")
add("inputs", "n_features_raw", train.shape[1] - len(targets) - len(excluded) - (1 if group else 0), "train")

if "metric" in val_metrics.columns:
    mcol = "metric"
elif "name" in val_metrics.columns:
    mcol = "name"
else:
    mcol = val_metrics.columns[0]
vcol = "value" if "value" in val_metrics.columns else val_metrics.columns[-1]
for _, r in val_metrics.head(10).iterrows():
    add("validation", r[mcol], r[vcol], "09_validate_score_validation_2WWEBTQD/validation_metrics.csv")

add("limitations", "no_refit", "Interpretation only; no model fitting performed.", "interpretation.csv")
add("limitations", "validation_only", "Metrics are from validation split; test not used.", "09_validate_score_validation_2WWEBTQD/validation_metrics.csv")
add("limitations", "top_feature", top_feature, "latest_model")

df = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"rows={len(df)} top_feature={top_feature}")
