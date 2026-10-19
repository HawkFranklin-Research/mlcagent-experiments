import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol)}")
task = protocol["task"]
if not isinstance(task, dict) or "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task) if isinstance(task, dict) else type(task)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    txt = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if txt and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        contiguous = (s.max() - s.min() + 1) == len(s)
        if contiguous or s.is_monotonic_increasing or s.is_monotonic_decreasing:
            continue
    feature_cols.append(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

alt_source = "08_model_train_alternative_3ESX6QWQ/alternative_model.pkl"
alt_manifest_source = "08_model_train_alternative_3ESX6QWQ/manifest.json"
pre_source = "05_preprocess_fit_preprocessing_C8Q2SNQ9/preprocessor.pkl"
sel_source = "06_features_retain_features_4S5ZAJJA/selector_all.pkl"
val_source = "09_validate_score_validation_JJ0ZPCSX/validation_metrics.csv"

alt = load_pkl(inputs[alt_source])
pre = load_pkl(inputs[pre_source])
sel = load_pkl(inputs[sel_source])
alt_manifest = json.loads(inputs[alt_manifest_source].read_text())

def final_est(obj):
    if isinstance(obj, dict):
        for k in ("model", "estimator", "pipeline"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, Pipeline):
        return obj.steps[-1][1]
    return obj

def walk_find(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == key:
                return v
            found = walk_find(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = walk_find(v, key)
            if found is not None:
                return found
    return None

def infer_top_feature(model, cols):
    est = final_est(model)
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_).ravel()
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        imp = np.abs(coef).sum(axis=0) if coef.ndim > 1 else np.abs(coef)
    if imp is None or imp.size == 0:
        return None
    names = None
    if isinstance(model, Pipeline):
        try:
            names = model[:-1].get_feature_names_out(cols)
        except Exception:
            names = None
    if names is None and getattr(est, "n_features_in_", None) == len(cols):
        names = cols
    if names is not None and len(names) == len(imp):
        return str(names[int(np.argmax(imp))])
    return None

top = walk_find(alt_manifest, "top_feature")
top_source = alt_manifest_source
if top is None:
    top = infer_top_feature(alt, feature_cols)
    top_source = alt_source
if top is None and feature_cols:
    top = feature_cols[0]
    top_source = alt_source

est = final_est(alt)
params = {}
try:
    for k, v in est.get_params(deep=False).items():
        if k != "estimator":
            params[k] = v
except Exception:
    pass

val = pd.read_csv(inputs[val_source])
val_items = []
if {"metric", "value"}.issubset(val.columns):
    for _, r in val.head(15).iterrows():
        val_items.append((str(r["metric"]), r["value"]))
else:
    for c in list(val.columns)[:10]:
        val_items.append((str(c), val[c].iloc[0] if len(val) else np.nan))

def fmt(v):
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.4g}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    return str(v)

rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": item, "value": fmt(value), "source": source})

add("model", "selected_model", alt_source, alt_source)
add("model", "final_estimator", type(est).__name__, alt_source)
steps = ", ".join([n for n, _ in alt.steps]) if isinstance(alt, Pipeline) else type(alt).__name__
add("model", "pipeline_steps", steps, alt_source)
add("model", "preprocessor", type(pre).__name__, pre_source)
add("model", "selector", type(sel).__name__, sel_source)
add("model", "top_feature", top, top_source)

for k, v in list(sorted(params.items()))[:10]:
    add("hyperparameters", k, v, alt_source)

add("inputs", "n_features", len(feature_cols), "train.parquet")
add("inputs", "feature_columns", ", ".join(map(str, feature_cols)), "train.parquet")
add("inputs", "targets", ", ".join(map(str, targets)), "train.parquet")
add("inputs", "excluded_columns", ", ".join(map(str, excluded)) if excluded else "(none)", "train.parquet")
add("inputs", "group_column", str(group) if group else "(none)", "train.parquet")
add("inputs", "primary_metric", str(primary), "train.parquet")

for k, v in val_items[:15]:
    add("validation", k, v, val_source)

add("limitations", "interpretation", "frozen model inspected without fitting or refitting", alt_source)
add("limitations", "validation", "evidence is from validation_metrics.csv only; no test exposure", val_source)
add("limitations", "scope", "hyperparameters are those saved in the frozen model; no search performed", alt_source)

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top, source=top_source)
print(len(feature_cols), len(targets), len(interp), top, primary)
