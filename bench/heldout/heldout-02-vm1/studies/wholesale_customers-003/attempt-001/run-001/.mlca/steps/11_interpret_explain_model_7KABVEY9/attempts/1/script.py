import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def load_json(p):
    return json.loads(p.read_text())

protocol = load_json(paths["_protocol"])
if "task" not in protocol:
    raise KeyError(f"_protocol missing 'task'; available keys: {sorted(protocol)}")
task = protocol["task"]
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
primary = task.get("primary_metric") or "silhouette"

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print(f"train_rows={len(train)} val_rows={len(val)}")

model = load_pkl(paths["latest_model"])
pre = load_pkl(paths["05_preprocess_fit_preprocessing_HBN850GP/preprocessor.pkl"])
sel = load_pkl(paths["06_features_retain_features_5BEJHTH5/selector_all.pkl"])

def final_est(obj):
    if isinstance(obj, Pipeline) and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    return obj

def fnames(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            n = fnames(step)
            if n:
                return n
    return None

final = final_est(model)
feat = fnames(final) or fnames(sel) or fnames(pre)
if not feat:
    drop = set(list(targets) + list(excluded) + ([group] if group else []))
    feat = [c for c in train.columns if c not in drop]
print(f"model={type(model).__name__} final={type(final).__name__} features={len(feat)}")

def top_feature(est, names):
    try:
        if hasattr(est, "feature_importances_"):
            score = np.abs(np.asarray(est.feature_importances_).ravel())
        elif hasattr(est, "coef_"):
            c = np.asarray(est.coef_)
            score = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c).ravel()
        elif hasattr(est, "cluster_centers_"):
            c = np.asarray(est.cluster_centers_)
            score = c.max(axis=0) - c.min(axis=0) if c.ndim == 2 else None
        else:
            return "unavailable"
        n = fnames(est) or names
        if score is not None and len(n) == len(score):
            return str(n[int(np.argmax(score))])
    except Exception:
        pass
    return "unavailable"

top = top_feature(final, feat)

def simple_params(est):
    try:
        p = est.get_params(deep=False)
    except Exception:
        return {}
    return {k: v for k, v in p.items() if isinstance(v, (str, int, float, bool, type(None)))}

params = simple_params(final)
param_s = "; ".join(f"{k}={v}" for k, v in list(params.items())[:12])

def sig(v):
    try:
        x = float(v)
        if np.isfinite(x):
            return float(f"{x:.4g}")
    except Exception:
        pass
    return v

vm = pd.DataFrame()
for key in ["09_validate_score_validation_Q3R7JTH4/validation_metrics.csv",
            "09_validate_score_validation_Q3R7JTH4/validation_metrics.parquet"]:
    p = paths[key]
    try:
        vm = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
        break
    except Exception:
        continue

metric_rows = []
if not vm.empty:
    low = {c.lower(): c for c in vm.columns}
    if "metric" in low and "value" in low:
        for _, r in vm.head(20).iterrows():
            metric_rows.append((str(r[low["metric"]]), sig(r[low["value"]])))
    else:
        for c in [c for c in vm.columns if pd.api.types.is_numeric_dtype(vm[c])][:20]:
            vals = vm[c].dropna()
            if len(vals):
                metric_rows.append((str(c), sig(vals.iloc[0] if len(vals) == 1 else vals.mean())))

rows = [
    ("task", "task_type", str(task.get("type", "")), "_protocol"),
    ("task", "primary_metric", str(primary), "_protocol"),
    ("model", "artifact", "latest_model", "latest_model"),
    ("model", "class", type(model).__name__, "latest_model"),
    ("model", "final_estimator", type(final).__name__, "latest_model"),
    ("model", "steps", " -> ".join(model.named_steps) if isinstance(model, Pipeline) else type(model).__name__, "latest_model"),
    ("model", "selected_hyperparameters", param_s, "latest_model"),
    ("inputs", "n_train_rows", len(train), "train"),
    ("inputs", "n_val_rows", len(val), "val"),
    ("inputs", "n_features", len(feat), "latest_model"),
    ("inputs", "feature_names", ", ".join(map(str, feat[:50])), "latest_model"),
    ("interpretation", "top_feature", top, "latest_model"),
]
for name, value in metric_rows:
    rows.append(("validation", name, value, "09_validate_score_validation_Q3R7JTH4/validation_metrics.csv"))
rows += [
    ("limitations", "no_outcome", "Clustering only; no ground-truth labels.", "_protocol"),
    ("limitations", "frozen", "No refit or threshold tuning in interpretation.", "latest_model"),
]

summary = pd.DataFrame(rows, columns=["section", "item", "value", "source"])
mlca_runtime.save_table(summary, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"top_feature={top}")
