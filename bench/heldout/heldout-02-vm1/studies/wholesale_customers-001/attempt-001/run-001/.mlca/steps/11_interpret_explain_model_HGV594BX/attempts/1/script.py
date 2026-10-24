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
        raise KeyError(f"Missing {path}; available top-level keys: {list(d.keys())}")
    return d[key]

protocol = json.loads(inputs["_protocol"].read_text())
targets = need(protocol, "targets", "targets")
excluded = need(protocol, "excluded_columns", "excluded_columns")
group = need(protocol, "group_column", "group_column")
task = need(protocol, "task", "task")
primary_metric = need(task, "primary_metric", "task.primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train rows={len(train)}, val rows={len(val)}")

def is_key_like(s):
    if s.dtype.kind in "O":
        return s.nunique(dropna=True) == len(train)
    if s.dtype.kind in "iu":
        n = s.name.lower()
        return any(t in n for t in ["id", "code", "key", "index", "seq"]) and s.nunique(dropna=True) == len(train)
    return False

group_set = set() if group is None else {group}
feature_cols = [
    c for c in train.columns
    if c not in set(targets) | set(excluded) | group_set and not is_key_like(train[c])
]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pkl(inputs["latest_model"])
prep = load_pkl(inputs["05_preprocess_fit_preprocessing_RXV573Z5/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_6C317WGV/selector_all.pkl"])

def final_est(m):
    return m.named_steps[m.steps[-1][0]] if isinstance(m, Pipeline) else m

fe = final_est(model)
def feat_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

model_features = feat_names(model) or feat_names(fe) or feature_cols
val_metric = None
try:
    vm = pd.read_csv(inputs["09_validate_score_validation_AHMWN36D/validation_metrics.csv"])
    if {"metric", "value"}.issubset(vm.columns):
        r = vm[vm["metric"].astype(str).str.lower() == primary_metric.lower()]
        if len(r):
            val_metric = float(r.iloc[0]["value"])
    else:
        for c in vm.columns:
            if c.lower() == primary_metric.lower():
                val_metric = float(vm.iloc[0][c])
                break
except Exception:
    val_metric = None

def top_feature_from(est, names):
    if hasattr(est, "cluster_centers_"):
        centers = np.asarray(est.cluster_centers_, dtype=float)
        if centers.ndim == 2 and centers.shape[1] == len(names):
            return names[int(np.argmax(centers.var(axis=0)))], "max variance of cluster_centers_ across clusters"
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if imp.shape[0] == len(names):
            return names[int(np.argmax(imp))], "max feature_importances_"
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        norm = np.linalg.norm(coef, axis=0) if coef.ndim == 2 else np.abs(coef)
        if norm.shape[0] == len(names):
            return names[int(np.argmax(norm))], "max coefficient norm"
    return None, "unavailable"

top_feature, top_method = top_feature_from(fe, model_features)
if top_feature is None and fe is not model:
    top_feature, top_method = top_feature_from(model, model_features)

def sig4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(f"{float(x):.4g}") if x != 0 else 0.0

def step_desc(obj):
    if isinstance(obj, Pipeline):
        return " > ".join(f"{n}:{type(s).__name__}" for n, s in obj.steps)
    return type(obj).__name__

rows = []
def add(item, text=None, num=None):
    rows.append({
        "item": item,
        "value_text": "" if text is None else str(text),
        "value_numeric": np.nan if num is None else sig4(num),
    })

add("model_type", type(fe).__name__)
add("model_pipeline", step_desc(model))
add("preprocessor_pipeline", step_desc(prep))
add("selector_pipeline", step_desc(sel))
add("n_input_features", len(model_features), len(model_features))
add("input_features", ", ".join(model_features))
add("raw_feature_schema", ", ".join(feature_cols))
add("validation_metric", primary_metric)
add("validation_value", val_metric, val_metric)
add("validation_split", "val")
add("limitations", (
    "Clustering has no outcome/label, so there is no external ground-truth validation. "
    "The cluster count and fit were chosen on training data; validation silhouette is an internal relative score only. "
    "Top feature is a descriptive frozen-model statistic, not a causal claim. "
    "Validation inputs are capped samples, so metrics may be less stable than on the full table."
))
add("top_feature", top_feature or "unavailable")
add("top_feature_method", top_method)
add("n_train_rows", len(train), len(train))
add("n_val_rows", len(val), len(val))

for k, v in sorted(getattr(fe, "get_params", lambda deep=False: {})(deep=False).items()):
    if len(rows) >= 50:
        break
    add(f"hyperparameter:{k}", v)

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature or "unavailable", source="interpretation.csv")
if val_metric is not None:
    mlca_runtime.record_metric(primary_metric, sig4(val_metric), "val")

print(f"model={type(fe).__name__}, features={len(model_features)}, top_feature={top_feature or 'unavailable'}")
print(f"val {primary_metric}={sig4(val_metric) if val_metric is not None else 'NA'}")
