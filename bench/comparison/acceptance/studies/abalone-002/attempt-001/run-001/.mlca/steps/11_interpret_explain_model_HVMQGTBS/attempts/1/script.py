import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key '{key}'. Available top-level keys: {list(d.keys())}")
    return d[key]

protocol = load_json(inputs["_protocol"])
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
group = req(protocol, "group_column")
primary_metric = task.get("primary_metric", "rmse")

train = pd.read_parquet(inputs["train"])
target_set = set(targets)
excluded_set = set(excluded)
group_set = {group} if group is not None else set()

candidates = [c for c in train.columns if c not in target_set and c not in excluded_set and c not in group_set]
key_like = []
for c in candidates:
    s = train[c]
    nun = s.nunique(dropna=False)
    if nun == len(s) and len(s) > 1:
        name_l = c.lower()
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            key_like.append(c)
        elif pd.api.types.is_integer_dtype(s):
            seq = s.dropna().sort_values()
            is_seq = len(seq) > 1 and seq.is_monotonic_increasing and (seq.diff().dropna() == 1).all()
            if is_seq or any(t in name_l for t in ("id", "key", "index", "row")):
                key_like.append(c)

feature_cols = [c for c in candidates if c not in key_like]

model = load_pickle(inputs["latest_model"])
alt_manifest = load_json(inputs["08_model_train_alternative_4WYQPWAG/manifest.json"])
pre_cls = "unavailable"
sel_cls = "unavailable"
try:
    pre_cls = type(load_pickle(inputs["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"])).__name__
except Exception:
    pass
try:
    sel_cls = type(load_pickle(inputs["06_features_retain_features_2K81ED4E/selector_all.pkl"])).__name__
except Exception:
    pass

def unwrap(est):
    for attr in ("regressor_", "estimator_", "best_estimator_", "model_"):
        if hasattr(est, attr):
            inner = getattr(est, attr)
            if inner is not None and inner is not est:
                return unwrap(inner)
    if hasattr(est, "estimators_") and len(getattr(est, "estimators_")) > 0:
        return unwrap(est.estimators_[0])
    return est

def final_estimator(m):
    if hasattr(m, "named_steps"):
        steps = list(m.named_steps.values())
        if steps:
            return unwrap(steps[-1])
    return unwrap(m)

def names_from(obj):
    if obj is None:
        return None
    if hasattr(obj, "get_feature_names_out"):
        try:
            return [str(x) for x in obj.get_feature_names_out()]
        except Exception:
            pass
    if hasattr(obj, "feature_names_in_"):
        try:
            return [str(x) for x in obj.feature_names_in_]
        except Exception:
            pass
    return None

def map_original(name):
    s = str(name)
    if s in feature_cols:
        return s
    for c in feature_cols:
        if c in s:
            return c
    if "__" in s:
        tail = s.split("__")[-1]
        for c in feature_cols:
            if c in tail:
                return c
    return s

est = final_estimator(model)
names = None
if hasattr(model, "named_steps"):
    steps = list(model.named_steps.values())
    for step in reversed(steps[:-1]):
        names = names_from(step)
        if names:
            break
if names is None:
    names = names_from(est)
if names is None:
    for key in ("06_features_retain_features_2K81ED4E/selector_all.pkl",
                "05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"):
        try:
            names = names_from(load_pickle(inputs[key]))
            if names:
                break
        except Exception:
            pass
if names is None:
    names = list(feature_cols)

importance = None
if hasattr(est, "feature_importances_"):
    try:
        importance = np.asarray(est.feature_importances_).ravel()
    except Exception:
        importance = None
if importance is None and hasattr(est, "coef_"):
    try:
        coef = np.asarray(est.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        importance = np.abs(coef).ravel()
    except Exception:
        importance = None

if importance is not None and len(importance) > 0:
    if len(importance) == len(names):
        top_raw = names[int(np.argmax(importance))]
    elif len(importance) == len(feature_cols):
        top_raw = feature_cols[int(np.argmax(importance))]
    elif len(names) > 0:
        top_raw = names[min(int(np.argmax(importance)), len(names) - 1)]
    else:
        top_raw = feature_cols[0] if feature_cols else "unknown"
else:
    top_raw = names[0] if names else (feature_cols[0] if feature_cols else "unknown")
top_feature = map_original(top_raw)

metrics_path = inputs["09_validate_score_validation_5RG1R1QC/validation_metrics.csv"]
vm = pd.read_csv(metrics_path)

def extract_metric(df, metric_name):
    ml = str(metric_name).lower()
    for col in df.columns:
        if str(col).lower() == ml:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    for _, row in df.iterrows():
        vals = [str(v).lower() for v in row.values]
        if any(ml == v or ml in v for v in vals):
            for v in row.values:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return float("nan")

val_metric = extract_metric(vm, primary_metric)

def find_key(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            found = find_key(v, keys)
            if found is not None:
                return found
    return None

hyper = find_key(alt_manifest, ["hyperparameters", "params", "best_params", "selected_hyperparameters"])
if hyper is None:
    try:
        hyper = est.get_params(deep=False)
    except Exception:
        hyper = {}
hyper_str = json.dumps(hyper, default=str) if hyper else "not declared"
if len(hyper_str) > 500:
    hyper_str = hyper_str[:497] + "..."

limitations = (
    "Frozen interpretation only; no fitting or refitting. Validation evidence is not test evidence. "
    "Feature importance/coefficient ranking can be unstable for correlated measurements; count outcome may be right-skewed."
)

rows = [
    {"item": "targets", "value": ", ".join(map(str, targets)), "source": inputs["_protocol"].name},
    {"item": "task_type", "value": task.get("type"), "source": inputs["_protocol"].name},
    {"item": "primary_metric", "value": primary_metric, "source": inputs["_protocol"].name},
    {"item": "frozen_model_artifact", "value": inputs["latest_model"].name, "source": inputs["latest_model"].name},
    {"item": "model_class", "value": type(model).__name__, "source": inputs["latest_model"].name},
    {"item": "selected_hyperparameters", "value": hyper_str, "source": inputs["08_model_train_alternative_4WYQPWAG/manifest.json"].name},
    {"item": "feature_count", "value": len(feature_cols), "source": inputs["train"].name},
    {"item": "feature_columns", "value": ", ".join(map(str, feature_cols)), "source": inputs["train"].name},
    {"item": "preprocessor", "value": pre_cls, "source": inputs["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"].name},
    {"item": "selector", "value": sel_cls, "source": inputs["06_features_retain_features_2K81ED4E/selector_all.pkl"].name},
    {"item": "validation_evidence", "value": metrics_path.name, "source": metrics_path.name},
    {"item": "validation_" + str(primary_metric), "value": val_metric, "source": metrics_path.name},
    {"item": "top_feature", "value": top_feature, "source": "interpretation.csv"},
    {"item": "limitations", "value": limitations, "source": inputs["_protocol"].name},
]

interp = pd.DataFrame(rows, columns=["item", "value", "source"])
interp["value"] = interp["value"].map(
    lambda x: f"{x:.4g}" if isinstance(x, (int, float, np.number)) and np.isfinite(x) else str(x)
)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation without fitting")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
if np.isfinite(val_metric):
    mlca_runtime.record_metric(str(primary_metric), float(val_metric), "val")

print(f"features: {len(feature_cols)}")
print(f"top_feature: {top_feature}")
if np.isfinite(val_metric):
    print(f"val {primary_metric}: {val_metric:.4g}")
print("interpretation.csv saved")
