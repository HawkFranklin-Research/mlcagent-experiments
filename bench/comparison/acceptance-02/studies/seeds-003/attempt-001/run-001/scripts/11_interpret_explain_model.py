import mlca_runtime
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def read_json(p):
    return json.loads(Path(p).read_text())

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def fmt(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.4g}"
    return str(v)

protocol = read_json(inp["_protocol"])
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]
key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        if s.notna().all() and s.nunique(dropna=True) == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

model_baseline = load_pkl(inp["07_model_train_baseline_9FVS2S2P/model.pkl"])
model_alt = load_pkl(inp["08_model_train_alternative_D2KDW4KS/alternative_model.pkl"])
model_latest = load_pkl(inp["latest_model"])
val_metrics = pd.read_csv(inp["09_validate_score_validation_R4H7PF4S/validation_metrics.csv"])

rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": item, "value": fmt(value), "source": source})

add("data", "train_rows", len(train), "train")
add("data", "val_rows", len(val), "val")
add("data", "n_features", len(feature_cols), "_protocol")
add("data", "targets", targets, "_protocol")
add("data", "excluded_columns", excluded, "_protocol")
add("data", "group_column", group if group is not None else "none", "_protocol")

def final_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model

def describe_model(label, model, source):
    add("model", f"{label}_class", type(model).__name__, source)
    final = final_estimator(model)
    add("model", f"{label}_final_estimator", type(final).__name__, source)
    if hasattr(final, "get_params"):
        params = final.get_params()
        for p in ["n_clusters", "init", "n_init", "max_iter", "random_state", "algorithm", "tol", "metric", "linkage"]:
            if p in params:
                add("hyperparameter", f"{label}.{p}", params[p], source)
    if hasattr(model, "named_steps"):
        add("model", f"{label}_steps", ",".join(model.named_steps.keys()), source)
    for obj in [model, final]:
        if hasattr(obj, "feature_names_in_"):
            fn = list(obj.feature_names_in_)
            preview = ", ".join(map(str, fn[:10]))
            if len(fn) > 10:
                preview += "..."
            add("input", f"{label}_feature_names", f"{len(fn)} columns: {preview}", source)
            break
    if hasattr(final, "cluster_centers_"):
        add("model", f"{label}_n_clusters", final.cluster_centers_.shape[0], source)
    if hasattr(final, "inertia_"):
        add("metric", f"{label}_inertia", float(final.inertia_), source)

describe_model("baseline", model_baseline, "07_model_train_baseline_9FVS2S2P/manifest.json")
describe_model("alternative", model_alt, "08_model_train_alternative_D2KDW4KS/manifest.json")
describe_model("latest", model_latest, "latest_model")

def extract_metric(df, name):
    d = df.copy()
    if "split" in d.columns:
        mask = d["split"].astype(str).str.lower().isin(["val", "validation", "valid"])
        if mask.any():
            d = d[mask]
    if name in d.columns:
        s = pd.to_numeric(d[name], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[0])
    for col in ["metric", "name", "metric_name"]:
        if col in d.columns and "value" in d.columns:
            sub = d[d[col].astype(str).str.lower() == name.lower()]
            if len(sub):
                s = pd.to_numeric(sub["value"], errors="coerce").dropna()
                if len(s):
                    return float(s.iloc[0])
    return None

metric_value = extract_metric(val_metrics, primary_metric)
add("validation", "primary_metric", primary_metric, "_protocol")
if metric_value is not None:
    add("validation", f"{primary_metric}_value", metric_value, "09_validate_score_validation_R4H7PF4S/validation_metrics.csv")
    mlca_runtime.record_metric(primary_metric, metric_value, "val")

add("limitation", "unsupervised", "No labelled outcome; grouping describes internal separation only.", "_protocol")
add("limitation", "generalisation", "Small seed-morphology table; clusters may not transfer to other populations.", "_protocol")
add("limitation", "refitting", "Frozen model was described without fitting; earlier selection is not re-evaluated here.", "latest_model")

def find_top_feature(model):
    final = final_estimator(model)
    feature_names = None
    try:
        feature_names = list(model.get_feature_names_out())
    except Exception:
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
    if hasattr(final, "cluster_centers_"):
        centers = np.asarray(final.cluster_centers_)
        if centers.ndim == 2 and centers.shape[1] > 0:
            idx = int(np.argmax(centers.var(axis=0)))
            if feature_names is not None and idx < len(feature_names):
                return str(feature_names[idx])
            return f"feature_{idx}"
    if hasattr(final, "feature_importances_"):
        imp = np.asarray(final.feature_importances_)
        if imp.ndim == 1:
            idx = int(np.argmax(imp))
            if feature_names is not None and idx < len(feature_names):
                return str(feature_names[idx])
            return f"feature_{idx}"
    return None

top_feature = find_top_feature(model_latest) or find_top_feature(model_alt) or find_top_feature(model_baseline)
if top_feature:
    add("feature", "top_feature", top_feature, "latest_model")
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

print(f"rows: train={len(train)}, val={len(val)}")
print(f"features: {len(feature_cols)}")
print("models described: baseline, alternative, latest")
if metric_value is not None:
    print(f"{primary_metric} val: {metric_value:.4f}")
print(f"interpretation rows: {len(interp)}")
