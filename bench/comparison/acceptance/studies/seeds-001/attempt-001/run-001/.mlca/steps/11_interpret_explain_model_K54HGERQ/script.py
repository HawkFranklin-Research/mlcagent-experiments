import json, pickle, re
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def read_json(path):
    return json.loads(path.read_text())

def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

protocol = read_json(inp["_protocol"])
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task key 'primary_metric'; available task keys: {list(task)}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
excluded = set(protocol.get("targets", [])) | set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
if group not in (None, ""):
    excluded.add(group)

feature_cols = []
for col in train.columns:
    if col in excluded:
        continue
    s = train[col]
    if s.dtype == object and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(col)

model_keys = [
    "07_model_train_baseline_FB68KDQ5/model.pkl",
    "08_model_train_alternative_A16CW8P2/alternative_model.pkl",
    "latest_model",
    "05_preprocess_fit_preprocessing_A5A2YZJ3/preprocessor.pkl",
    "06_features_retain_features_297WVMMN/selector_all.pkl",
]
models = {k: read_pkl(inp[k]) for k in model_keys if k in inp}
latest_key = "latest_model" if "latest_model" in inp else "08_model_train_alternative_A16CW8P2/alternative_model.pkl"
latest = models.get(latest_key)
latest_source = "08_model_train_alternative_A16CW8P2/alternative_model.pkl" if "08_model_train_alternative_A16CW8P2/alternative_model.pkl" in inp else latest_key

def unwrap(obj):
    if isinstance(obj, dict):
        for key in ("model", "estimator", "pipeline", "best_estimator_"):
            if key in obj:
                return unwrap(obj[key])
    if hasattr(obj, "best_estimator_"):
        return unwrap(obj.best_estimator_)
    return obj

def final_est(obj):
    obj = unwrap(obj)
    if isinstance(obj, Pipeline):
        return unwrap(obj.named_steps[list(obj.named_steps)[-1]])
    return obj

def feature_importance(obj):
    est = final_est(obj)
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float))
        imp = imp.mean(axis=0) if imp.ndim > 1 else imp.ravel()
    elif hasattr(est, "cluster_centers_"):
        imp = np.asarray(est.cluster_centers_, dtype=float).var(axis=0)
    if imp is None or len(imp) == 0 or np.all(pd.isna(imp)):
        return None, None
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    names = list(est.feature_names_in_) if hasattr(est, "feature_names_in_") else None
    if names is None or len(names) != len(imp):
        names = feature_cols if len(feature_cols) == len(imp) else [f"feature_{i}" for i in range(len(imp))]
    names = [re.split(r"__", str(n))[-1] for n in names]
    idx = int(np.argmax(imp))
    return names[idx], float(imp[idx])

top_feature, top_score = feature_importance(latest) if latest is not None else (None, None)

vm = pd.read_csv(inp["09_validate_score_validation_SVSGG8S9/validation_metrics.csv"])
num_cols = list(vm.select_dtypes(include=[np.number]).columns)
str_cols = [c for c in vm.columns if c not in num_cols]
metric_col = next((c for c in vm.columns if str(c).lower() in ("metric", "name", "metric_name", "measure")), str_cols[0] if str_cols else None)
val_col = next((c for c in num_cols if str(c).lower() in ("value", "score", "metric_value", "val")), num_cols[0] if num_cols else None)

rows = []
def add(section, item, value, source):
    if value is None:
        return
    text = str(value)
    if len(text) > 500:
        text = text[:497] + "..."
    rows.append({"section": section, "item": item, "value": text, "source": str(source)})

add("model", "selected_model", type(final_est(latest)).__name__ if latest is not None else "missing", latest_source)
if latest is not None:
    est = final_est(latest)
    try:
        params = est.get_params(deep=False)
    except Exception:
        params = {}
    keep = {k: v for k, v in params.items() if k in ("n_clusters", "init", "n_init", "max_iter", "tol", "random_state", "covariance_type", "reg_covar", "metric", "linkage", "distance_threshold", "eps", "min_samples", "bandwidth")}
    if not keep:
        keep = {k: v for k, v in params.items() if not k.endswith("_")}
    add("hyperparameters", "estimator_params", json.dumps(keep, default=str, sort_keys=True), latest_source)

for key, label in [
    ("07_model_train_baseline_FB68KDQ5/manifest.json", "baseline_manifest"),
    ("08_model_train_alternative_A16CW8P2/manifest.json", "alternative_manifest"),
    ("09_validate_score_validation_SVSGG8S9/manifest.json", "validation_manifest"),
]:
    if key in inp:
        try:
            manifest = read_json(inp[key])
            small = {k: manifest[k] for k in ("selected_model", "model", "estimator", "hyperparameters", "params", "metrics", "primary_metric", "n_clusters", "feature_names", "best_score") if k in manifest}
            if not small:
                small = {k: manifest[k] for k in list(manifest)[:10]}
            add("manifest", label, json.dumps(small, default=str, sort_keys=True), key)
        except Exception as exc:
            add("manifest", label, f"unreadable: {type(exc).__name__}", key)

if val_col is not None and metric_col is not None:
    for _, row in vm.head(10).iterrows():
        name, value = row[metric_col], row[val_col]
        if pd.notna(value):
            add("validation", f"metric:{name}", f"{float(value):.4g}", "09_validate_score_validation_SVSGG8S9/validation_metrics.csv")
            if str(name) == str(primary_metric):
                try:
                    mlca_runtime.record_metric(str(primary_metric), float(value), "val")
                except Exception:
                    pass
else:
    add("validation", "primary_metric", primary_metric, "09_validate_score_validation_SVSGG8S9/validation_metrics.csv")

add("inputs", "n_train_rows", len(train), "train.parquet")
add("inputs", "n_feature_columns", len(feature_cols), "train.parquet")
add("inputs", "feature_columns", ", ".join(map(str, feature_cols[:20])) + ("..." if len(feature_cols) > 20 else ""), "train.parquet")
add("inputs", "protocol_primary_metric", primary_metric, "_protocol")
add("inputs", "protocol_targets", json.dumps(protocol.get("targets", [])), "_protocol")
add("inputs", "protocol_excluded_columns", json.dumps(protocol.get("excluded_columns", [])), "_protocol")
add("inputs", "protocol_group_column", protocol.get("group_column"), "_protocol")

for key, text in [
    ("no_outcome", "No outcome/label column exists; the model is unsupervised over kernel geometry measurements."),
    ("cluster_count", "Cluster count is chosen from internal criteria only, not external labels."),
    ("silhouette", "Silhouette is internal and scale-dependent; it does not by itself establish semantic kernel classes."),
    ("scope", "Interpretation describes a frozen model without refitting or test exposure."),
    ("generalization", "Validation evidence comes from the provided split; stability on new kernel batches is untested."),
]:
    add("limitations", key, text, "_protocol")

if top_feature is not None:
    add("model", "top_feature", top_feature, latest_source)
    add("model", "top_feature_score", f"{top_score:.4g}", latest_source)
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source=latest_source)
else:
    add("model", "top_feature", "unavailable", latest_source)
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source=latest_source)

interpretation = pd.DataFrame(rows)[["section", "item", "value", "source"]].head(50)
mlca_runtime.save_table(interpretation, "interpretation.csv", caption="Frozen clustering model interpretation")

print(f"interpretation rows: {len(interpretation)}")
print(f"selected model: {type(final_est(latest)).__name__ if latest is not None else 'missing'}")
print(f"top_feature: {top_feature}")
print(f"primary_metric: {primary_metric}")
