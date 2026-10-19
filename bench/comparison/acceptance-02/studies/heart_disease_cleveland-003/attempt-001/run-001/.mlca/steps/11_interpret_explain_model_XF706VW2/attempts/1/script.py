import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
if "group_column" not in protocol:
    raise KeyError(f"_protocol missing 'group_column'; available top-level keys: {sorted(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
ex = set(targets) | set(excluded)
if group is not None:
    ex.add(group)
key_cols = []
for c in train.columns:
    if c in ex:
        continue
    s = train[c]
    if len(train) > 1 and s.nunique(dropna=False) == len(train):
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_integer_dtype(s)):
            key_cols.append(c)
features = [c for c in train.columns if c not in ex and c not in key_cols]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def final_est(m):
    if isinstance(m, Pipeline):
        return m.named_steps[list(m.named_steps)[-1]]
    return m

def fmt(v):
    if isinstance(v, (float, int, np.number)) and not isinstance(v, bool):
        return f"{float(v):.4g}"
    return str(v)

def top_feature(m, names):
    est = final_est(m)
    vals = None
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_).ravel()
    elif hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        vals = np.abs(c).mean(axis=0).ravel() if c.ndim > 1 else np.abs(c).ravel()
    elif hasattr(est, "estimators_"):
        arrs = []
        for e in est.estimators_:
            if hasattr(e, "coef_"):
                arrs.append(np.abs(np.asarray(e.coef_)).ravel())
            elif hasattr(e, "feature_importances_"):
                arrs.append(np.asarray(e.feature_importances_).ravel())
        if arrs:
            vals = np.mean(arrs, axis=0)
    if vals is None or len(vals) == 0:
        return "unavailable"
    idx = int(np.argmax(np.abs(vals)))
    return str(names[idx]) if idx < len(names) else f"feature_index_{idx}"

base = load_pkl(inp["07_model_train_baseline_X4GJNHE4/model.pkl"])
latest = load_pkl(inp["latest_model"])
try:
    selector = load_pkl(inp["06_features_retain_features_NH52E6T3/selector_all.pkl"])
except Exception:
    selector = None

names = features
if selector is not None and hasattr(selector, "get_support") and hasattr(selector, "feature_names_in_"):
    try:
        names = list(selector.feature_names_in_[selector.get_support()])
    except Exception:
        pass
tf = top_feature(latest, names)

rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": item, "value": fmt(value), "source": source})

add("model", "baseline_class", type(base).__name__, "07_model_train_baseline_X4GJNHE4/model.pkl")
add("model", "latest_model_key", "latest_model", "latest_model")
add("model", "latest_class", type(latest).__name__, "latest_model")
if isinstance(latest, Pipeline):
    add("model", "latest_steps", " -> ".join(latest.named_steps.keys()), "latest_model")
est = final_est(latest)
add("model", "latest_estimator", type(est).__name__, "latest_model")
add("model", "latest_hyperparameters", json.dumps(est.get_params(), default=str)[:800], "latest_model")
add("inputs", "targets", ", ".join(map(str, targets)), "_protocol")
add("inputs", "excluded_columns", ", ".join(map(str, excluded)) if excluded else "none", "_protocol")
add("inputs", "group_column", str(group), "_protocol")
add("inputs", "n_features", len(features), "train.parquet")
for f in features:
    add("inputs", f"feature:{f}", str(train[f].dtype), "train.parquet")
add("inputs", "key_like_excluded", ", ".join(key_cols) if key_cols else "none", "train.parquet")
add("validation", "primary_metric", primary_metric, "_protocol")
try:
    vm = pd.read_csv(inp["09_validate_score_validation_QG0J48N2/validation_metrics.csv"])
    vsource = "09_validate_score_validation_QG0J48N2/validation_metrics.csv"
except Exception:
    vm = pd.read_parquet(inp["09_validate_score_validation_QG0J48N2/validation_metrics.parquet"])
    vsource = "09_validate_score_validation_QG0J48N2/validation_metrics.parquet"
nmetrics = 0
if "metric" in vm.columns and "value" in vm.columns:
    for _, r in vm.head(20).iterrows():
        add("validation", str(r["metric"]), r["value"], vsource)
        nmetrics += 1
else:
    for c in list(vm.columns)[:12]:
        add("validation", str(c), vm[c].iloc[0] if len(vm) else np.nan, vsource)
        nmetrics += 1
add("limitations", "no_refit", "Loaded frozen objects; no fitting or refitting was performed.", "latest_model")
add("limitations", "validation_only", "Evidence is validation-only; test data was not used.", vsource)
add("limitations", "target_dependency", "heart_disease_binary is a deterministic coarsening of num; both are joint targets.", "_protocol")
add("limitations", "cohort_size", "Cleveland cohort is modest; validation estimates may be unstable.", "train.parquet")
add("facts", "top_feature", tf, "interpretation.csv")

df = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen-model interpretation: model, hyperparameters, inputs, validation evidence and limitations.")
mlca_runtime.record_fact_candidate("top_feature", tf, source="interpretation.csv")
print("interpretation rows:", len(df))
print("top_feature:", tf)
print("validation metrics:", nmetrics)
