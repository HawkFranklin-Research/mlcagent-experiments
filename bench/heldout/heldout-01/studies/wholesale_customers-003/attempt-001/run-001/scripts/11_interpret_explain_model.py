import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"] or []
excluded = protocol["excluded_columns"] or []
group = protocol["group_column"]
task = protocol["task"]
primary_metric = task.get("primary_metric")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s, name):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        return any(tok in str(name).lower() for tok in ["id", "key", "index", "row"])
    return False

drop = set(targets) | set(excluded) | ({group} if group is not None else set())
features = [c for c in train.columns if c not in drop and not is_key_like(train[c], c)]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

selected_model = load_pkl("latest_model")
baseline_model = load_pkl("07_model_train_baseline_4QYJGXS4/model.pkl")

def final_est(obj):
    if isinstance(obj, Pipeline) and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    if hasattr(obj, "steps") and obj.steps:
        return obj.steps[-1][1]
    return obj

def fmt_params(est):
    try:
        params = est.get_params(deep=False)
    except Exception:
        return ""
    keep = [f"{k}={v}" for k, v in sorted(params.items())
            if isinstance(v, (str, int, float, bool, type(None)))]
    return "; ".join(keep[:12])

def model_type(m):
    if isinstance(m, Pipeline) and m.named_steps:
        return "Pipeline(" + " -> ".join(f"{k}:{type(v).__name__}" for k, v in m.named_steps.items()) + ")"
    return type(final_est(m)).__name__

def top_feature_from_model(m):
    est = final_est(m)
    names = None
    for obj in [est, m]:
        if hasattr(obj, "feature_names_in_"):
            names = list(obj.feature_names_in_)
            break
    if names is None:
        return None
    n = len(names)
    if hasattr(est, "feature_importances_") and len(est.feature_importances_) == n:
        return names[int(np.argmax(est.feature_importances_))]
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        vals = np.abs(coef) if coef.ndim == 1 else np.abs(coef).sum(axis=0)
        if len(vals) == n:
            return names[int(np.argmax(vals))]
    if hasattr(est, "cluster_centers_"):
        centers = np.asarray(est.cluster_centers_)
        if centers.ndim == 2 and centers.shape[1] == n:
            return names[int(np.argmax(np.var(centers, axis=0)))]
    return None

top_feature = top_feature_from_model(selected_model) or top_feature_from_model(baseline_model)
if top_feature is None:
    variances = {c: float(train[c].var(numeric_only=True)) for c in features
                 if pd.api.types.is_numeric_dtype(train[c])}
    top_feature = max(variances, key=variances.get) if variances else (features[0] if features else "NA")

metrics = pd.read_csv(inp["09_validate_score_validation_BJKNFAPK/validation_metrics.csv"])
def first_numeric(needle):
    for c in metrics.columns:
        if needle in c.lower():
            s = metrics[c].dropna()
            if len(s) and pd.api.types.is_numeric_dtype(s):
                return float(s.iloc[0])
    return None

val_sil = first_numeric("silhouette")
sel_k = None
for c in metrics.columns:
    if c.lower() in ("k", "n_clusters", "cluster_count"):
        s = metrics[c].dropna()
        if len(s):
            sel_k = int(s.iloc[0])

model_path = "08_model_train_alternative_W0P9YC9A/alternative_model.pkl"
rows = [
    ("model", model_type(selected_model), model_path),
    ("selected_hyperparameters", fmt_params(final_est(selected_model)) or "not available", model_path),
    ("feature_count", str(len(features)), "train"),
    ("feature_columns", ", ".join(features), "train"),
    ("train_rows", str(len(train)), "train"),
    ("val_rows", str(len(val)), "val"),
    ("primary_metric", str(primary_metric), "_protocol"),
    ("top_feature", top_feature, "interpretation.csv"),
    ("limitation_1", "No outcome or label column exists; interpretation relies on internal silhouette evidence only.", "_protocol"),
    ("limitation_2", "Frozen model is described without refitting; selected hyperparameters and clusters are as saved.", model_path),
    ("limitation_3", "Silhouette is sensitive to scaling and distance metric; cluster labels are not externally validated.", "09_validate_score_validation_BJKNFAPK/validation_metrics.csv"),
]
if val_sil is not None:
    rows.append(("validation_silhouette", f"{val_sil:.4g}", "09_validate_score_validation_BJKNFAPK/validation_metrics.csv"))
if sel_k is not None:
    rows.append(("selected_clusters", str(sel_k), "09_validate_score_validation_BJKNFAPK/validation_metrics.csv"))

interp = pd.DataFrame(rows, columns=["aspect", "detail", "source_file"]).iloc[:50]
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print("model:", model_type(selected_model))
print("rows:", len(train), "train /", len(val), "val")
print("features:", len(features))
print("top_feature:", top_feature)
print("validation_silhouette:", "NA" if val_sil is None else f"{val_sil:.4g}")
