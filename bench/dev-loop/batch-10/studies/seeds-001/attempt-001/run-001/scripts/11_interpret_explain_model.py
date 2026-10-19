import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

def opt(d, k, default=None):
    return d.get(k, default)

protocol = json.loads(inp["_protocol"].read_text())
targets = list(opt(protocol, "targets", []) or [])
excluded = list(opt(protocol, "excluded_columns", []) or [])
group_col = opt(protocol, "group_column", None)
task = opt(protocol, "task", {}) or {}
primary_metric = task.get("primary_metric", "reconstruction_error") if isinstance(task, dict) else "reconstruction_error"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl(inp["05_preprocess_fit_preprocessing_6DN7THMT/preprocessor.pkl"])
selector = load_pkl(inp["06_features_retain_features_AKTAEF6F/selector_all.pkl"])
baseline = load_pkl(inp["07_model_train_baseline_V7ETXH3B/model.pkl"])
latest_model = load_pkl(inp["latest_model"])
vm = pd.read_csv(inp["09_validate_score_validation_GVTVRD4E/validation_metrics.csv"])

def is_keylike(s):
    if s.notna().all() and s.nunique(dropna=True) == len(s):
        if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
            return True
        if pd.api.types.is_integer_dtype(s.dtype):
            return True
    return False

skip = set(targets) | set(excluded)
if group_col is not None:
    skip.add(group_col)
feature_cols = [c for c in train.columns if c not in skip and not is_keylike(train[c])]

def safe_names(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is None:
        return []
    try:
        return [str(x) for x in list(fn)]
    except Exception:
        return []

model_features = safe_names(latest_model)
if not model_features:
    model_features = list(feature_cols)
features_match = len(model_features) == len(feature_cols) and set(model_features) == set(feature_cols)

def steps_of(obj):
    if isinstance(obj, Pipeline):
        return list(obj.named_steps.keys())
    return [type(obj).__name__]

def short_params(est):
    keep = {"n_components", "svd_solver", "whiten", "hidden_layer_sizes", "activation",
            "alpha", "learning_rate_init", "max_iter", "tol", "random_state", "solver"}
    try:
        params = est.get_params(deep=False)
    except Exception:
        return ""
    selected = {k: v for k, v in params.items() if k in keep and v is not None}
    return "; ".join(f"{k}={v}" for k, v in sorted(selected.items())[:8])

def final_est(model):
    if isinstance(model, Pipeline) and len(model.steps) > 0:
        return model.steps[-1][1]
    return model

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None

def extract_top_feature(model):
    names = safe_names(model)
    est = final_est(model)
    if not names:
        nfi = getattr(est, "n_features_in_", None)
        if nfi is not None and int(nfi) == len(feature_cols):
            names = list(feature_cols)
    if not names:
        return None
    nfi = getattr(est, "n_features_in_", None)
    if nfi is not None and int(nfi) != len(names):
        return None
    for attr in ("feature_importances_", "components_", "coefs_", "coef_"):
        if hasattr(est, attr):
            try:
                w = np.abs(np.asarray(getattr(est, attr)))
            except Exception:
                continue
            if attr == "components_":
                if w.ndim == 2 and w.shape[1] == len(names):
                    return names[int(np.argmax(w.sum(axis=0)))]
            elif attr == "coefs_":
                if w.ndim == 2 and w.shape[0] == len(names):
                    return names[int(np.argmax(w.sum(axis=1)))]
            else:
                w = w.ravel()
                if len(w) == len(names):
                    return names[int(np.argmax(w))]
    return None

manifest_sources = [
    ("07_model_train_baseline_V7ETXH3B/manifest.json", inp["07_model_train_baseline_V7ETXH3B/manifest.json"]),
    ("08_model_train_alternative_G1KJG23J/manifest.json", inp["08_model_train_alternative_G1KJG23J/manifest.json"]),
]
top_feature = None
top_source = "latest_model"
for src_name, src_path in manifest_sources:
    try:
        js = json.loads(src_path.read_text())
        val_js = find_key(js, "top_feature")
        if val_js is not None:
            top_feature = str(val_js)
            top_source = src_name
            break
    except Exception:
        pass
if top_feature is None:
    top_feature = extract_top_feature(latest_model)
if top_feature is None:
    top_feature = "unavailable"

primary_value = None
metric_col = next((c for c in vm.columns if str(c).lower() in {"metric", "name", "key", "metric_name"}), None)
value_col = next((c for c in vm.columns if str(c).lower() in {"value", "score", "mean", "metric_value"}), None)
if primary_metric in vm.columns:
    vals = pd.to_numeric(vm[primary_metric], errors="coerce").dropna()
    if len(vals):
        primary_value = float(vals.iloc[0])
if primary_value is None and metric_col is not None and value_col is not None:
    mask = vm[metric_col].astype(str).str.contains(primary_metric, case=False, regex=False)
    if mask.any():
        vals = pd.to_numeric(vm.loc[mask, value_col], errors="coerce").dropna()
        if len(vals):
            primary_value = float(vals.iloc[0])

rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": item, "value": str(value), "source": source})

add("model", "selected_model", type(latest_model).__name__, "latest_model")
add("model", "pipeline_steps", ", ".join(steps_of(latest_model)), "latest_model")
add("model", "final_estimator", type(final_est(latest_model)).__name__, "latest_model")
add("model", "selected_hyperparameters", short_params(final_est(latest_model)) or "defaults", "latest_model")
add("inputs", "train_rows", len(train), "train.parquet")
add("inputs", "val_rows", len(val), "val.parquet")
add("inputs", "feature_count", len(feature_cols), "train.parquet")
add("inputs", "model_features_match_protocol", bool(features_match), "train.parquet")
add("inputs", "feature_columns", ", ".join(map(str, feature_cols[:12])) + ("..." if len(feature_cols) > 12 else ""), "train.parquet")
add("preprocessing", "preprocessor_steps", ", ".join(steps_of(preprocessor)), "05_preprocess_fit_preprocessing_6DN7THMT/preprocessor.pkl")
add("preprocessing", "selector_steps", ", ".join(steps_of(selector)), "06_features_retain_features_AKTAEF6F/selector_all.pkl")
add("validation", "primary_metric", primary_metric, "09_validate_score_validation_GVTVRD4E/validation_metrics.csv")
if primary_value is not None:
    add("validation", "primary_metric_value", round(primary_value, 4), "09_validate_score_validation_GVTVRD4E/validation_metrics.csv")
if metric_col is not None and value_col is not None:
    for _, r in vm.head(6).iterrows():
        add("validation", str(r[metric_col]), r[value_col], "09_validate_score_validation_GVTVRD4E/validation_metrics.csv")
add("limitations", "outcome", "No outcome column; representation learning only.", "_protocol")
add("limitations", "fitting", "Frozen models described only; no refit after validation exposure.", "latest_model")
add("limitations", "validation_scope", "Reconstruction evidence is in the frozen frontend feature space; inputs are capped samples.", "09_validate_score_validation_GVTVRD4E/validation_metrics.csv")
add("fact", "top_feature", top_feature, top_source)

interp = pd.DataFrame(rows, columns=["section", "item", "value", "source"])

mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen representation model interpretation")
if primary_value is not None:
    mlca_runtime.record_metric(primary_metric, primary_value, split="val")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"features={len(feature_cols)} model_features={len(model_features)}")
print(f"model={type(latest_model).__name__} estimator={type(final_est(latest_model)).__name__}")
print(f"{primary_metric}={primary_value if primary_value is not None else 'unavailable'}")
print("saved=interpretation.csv")
