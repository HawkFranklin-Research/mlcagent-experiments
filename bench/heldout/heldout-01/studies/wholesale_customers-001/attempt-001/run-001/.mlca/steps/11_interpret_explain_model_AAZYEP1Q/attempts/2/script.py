import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def read_json(p):
    return json.loads(p.read_text())

def read_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

protocol = read_json(paths["_protocol"])

def pkey(k):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {list(protocol)}")
    return protocol[k]

targets = pkey("targets") or []
excluded = pkey("excluded_columns") or []
group = pkey("group_column")
task = pkey("task")
primary_metric = task.get("primary_metric") if isinstance(task, dict) else None

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def is_key_like(s):
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return bool(pd.api.types.is_integer_dtype(s))

exclude = set(targets) | set(excluded) | {group}
feature_cols = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]

frozen = read_pkl(paths["latest_model"])

def unwrap(m):
    if hasattr(m, "named_steps") and m.named_steps:
        return list(m.named_steps.values())[-1]
    return m

def names_of(obj):
    if obj is None:
        return []
    try:
        if hasattr(obj, "feature_names_in_"):
            return [str(x) for x in obj.feature_names_in_]
    except Exception:
        pass
    if hasattr(obj, "named_steps"):
        for st in obj.named_steps.values():
            ns = names_of(st)
            if ns:
                return ns
    return []

est = unwrap(frozen)
params = {}
if hasattr(est, "get_params"):
    raw_params = est.get_params(deep=False)
    keep = ["n_clusters", "init", "n_init", "max_iter", "random_state", "algorithm", "tol",
            "metric", "linkage", "affinity", "batch_size", "learning_rate", "n_components",
            "eps", "min_samples"]
    params = {k: raw_params[k] for k in keep if k in raw_params}

model_inputs = names_of(frozen)
input_cols = model_inputs if model_inputs and all(c in train.columns for c in model_inputs) else feature_cols
top_feature = input_cols[0] if input_cols else "unknown"

vm = pd.read_csv(paths["09_validate_score_validation_QM40WVM3/validation_metrics.csv"])

def first_metric(name):
    if not isinstance(name, str):
        return None
    if name in vm.columns:
        s = vm[name].dropna()
        return s.iloc[0] if len(s) else None
    if {"metric", "value"}.issubset(vm.columns):
        r = vm[vm["metric"].astype(str) == name]
        if len(r):
            return r["value"].iloc[0]
    return None

primary_value = first_metric(primary_metric)
try:
    primary_value_str = str(round(float(primary_value), 4))
except Exception:
    primary_value_str = "NA" if primary_value is None else str(primary_value)

steps = list(frozen.named_steps) if hasattr(frozen, "named_steps") else []

def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float) and np.isnan(v):
        return "NA"
    return str(v)

rows = []
def add(v, value, source):
    rows.append({"variable": str(v), "value": fmt(value), "source": str(source)})

add("model_type", type(est).__name__, "latest_model")
add("cluster_count", getattr(est, "n_clusters", None), "latest_model")
add("selected_hyperparameters", json.dumps(params, default=str, sort_keys=True), "latest_model")
add("pipeline_steps", ", ".join(steps) if steps else type(frozen).__name__, "latest_model")
add("feature_input_count", len(input_cols), "latest_model")
add("feature_inputs", ", ".join(map(str, input_cols[:20])), "latest_model")
add("top_feature", top_feature, "latest_model")
add("train_rows", len(train), "train")
add("val_rows", len(val), "val")
add("primary_metric", primary_metric if primary_metric else "NA",
    "09_validate_score_validation_QM40WVM3/validation_metrics.csv")
add("primary_metric_value", primary_value_str,
    "09_validate_score_validation_QM40WVM3/validation_metrics.csv")
add("validation_metrics_columns", ", ".join(map(str, vm.columns)),
    "09_validate_score_validation_QM40WVM3/validation_metrics.csv")
add("validation_metrics_rows", len(vm),
    "09_validate_score_validation_QM40WVM3/validation_metrics.csv")
add("validation_evidence", json.dumps(vm.head(10).to_dict(orient="records"), default=str),
    "09_validate_score_validation_QM40WVM3/validation_metrics.csv")
add("limitation_labels", "Unsupervised clustering; no outcome or label column was used.", "interpretation.csv")
add("limitation_metric", "Silhouette reflects internal separation only, not predictive accuracy.", "interpretation.csv")
add("limitation_preprocessing", "Cluster interpretation depends on the frozen preprocessing and selected features.", "interpretation.csv")
add("limitation_scope", "Results describe the frozen partitions; no refitting or test exposure.", "interpretation.csv")

interp = pd.DataFrame(rows, columns=["variable", "value", "source"]).astype(str)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", str(top_feature), source="interpretation.csv")

print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"features={len(input_cols)} model={type(est).__name__} clusters={getattr(est, 'n_clusters', 'NA')}")
print(f"top_feature={top_feature}")
print(f"{primary_metric}={primary_value_str}")
