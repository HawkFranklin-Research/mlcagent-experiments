import mlca_runtime, json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ["task", "targets", "excluded_columns"]:
    if k not in protocol:
        raise KeyError(f"_protocol missing {k}; available keys={list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"task missing primary_metric; available keys={list(task.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
metrics_path = inputs["09_validate_score_validation_51H7Y4ST/validation_metrics.csv"]
metrics = pd.read_csv(metrics_path)
metrics_source = "09_validate_score_validation_51H7Y4ST/validation_metrics.csv"

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

model = load_pkl("latest_model")
baseline = load_pkl("07_model_train_baseline_VKCYA88G/model.pkl")
alt = load_pkl("08_model_train_alternative_65RN6RC0/alternative_model.pkl")
model_source = "08_model_train_alternative_65RN6RC0/alternative_model.pkl"
baseline_source = "07_model_train_baseline_VKCYA88G/model.pkl"

def final_est(m):
    obj = m
    while hasattr(obj, "named_steps"):
        obj = list(obj.named_steps.values())[-1]
    return obj

def model_type_str(m):
    fe = final_est(m)
    if hasattr(m, "named_steps"):
        steps = ",".join(map(str, m.named_steps.keys()))
        return f"{type(m).__name__}[{steps}] -> {type(fe).__name__}"
    return type(fe).__name__

def hp_str(m):
    fe = final_est(m)
    p = fe.get_params(deep=False)
    keys = []
    terms = ["max_depth", "n_estimators", "learning_rate", "alpha", "solver",
             "loss", "penalty", "min_samples", "subsample", "l2_regularization",
             "max_iter", "n_neighbors", "kernel", "gamma", "max_leaf_nodes",
             "min_samples_leaf", "c"]
    for k in p:
        if any(s in k.lower() for s in terms):
            keys.append(k)
    if not keys:
        keys = [k for k in p if k != "random_state"][:8]
    return "; ".join(f"{k}={p[k]}" for k in keys[:8])

def fnames(m):
    if hasattr(m, "feature_names_in_"):
        return [str(x) for x in m.feature_names_in_]
    if hasattr(m, "named_steps"):
        for s in m.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return [str(x) for x in s.feature_names_in_]
    fe = final_est(m)
    if hasattr(fe, "feature_names_in_"):
        return [str(x) for x in fe.feature_names_in_]
    return []

def top_feature(m):
    fe = final_est(m)
    vals = None
    if hasattr(fe, "coef_"):
        vals = np.ravel(fe.coef_)
    elif hasattr(fe, "feature_importances_"):
        vals = np.ravel(fe.feature_importances_)
    if vals is None or len(vals) == 0:
        return None
    names = getattr(fe, "feature_names_in_", None)
    if names is None or len(names) != len(vals):
        names = getattr(m, "feature_names_in_", None)
    if names is None or len(names) != len(vals):
        return None
    return str(list(names)[int(np.argmax(np.abs(vals)))])

def metric_value(df, name):
    sub = df
    if "metric" in df.columns:
        mask = df["metric"].astype(str).str.lower().str.contains(str(name).lower(), na=False)
        if mask.any():
            sub = df[mask]
    for c in ["value", "score", "mean", "metric_value", "validation"]:
        if c in sub.columns:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[0])
    for c in sub.columns:
        v = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(v):
            return float(v.iloc[0])
    return None

val_metric = metric_value(metrics, primary)
if val_metric is not None:
    mlca_runtime.record_metric(f"validation_{primary}", val_metric, "val")

tf, tf_src = None, ""
for obj, src in [(model, model_source), (baseline, baseline_source), (alt, model_source)]:
    t = top_feature(obj)
    if t:
        tf, tf_src = t, src
        break
if tf:
    mlca_runtime.record_fact_candidate("top_feature", tf, source=tf_src)

records = []
def add(a, d, s=""):
    records.append({"aspect": str(a), "detail": str(d)[:500], "source": str(s)})

add("primary_metric", primary, "_protocol")
add("targets", ", ".join(map(str, targets)), "_protocol")
add("excluded_columns", ", ".join(map(str, excluded)), "_protocol")
add("task_type", task.get("type", ""), "_protocol")
add("train_rows", len(train), "train")
add("val_rows", len(val), "val")
add("frozen_model_type", model_type_str(model), model_source)
add("selected_hyperparameters", hp_str(model), model_source)
fn = fnames(model)
add("model_input_feature_count", len(fn), model_source)
if fn:
    add("model_input_features_preview", ", ".join(fn[:12]) + (" ..." if len(fn) > 12 else ""), model_source)
add("validation_metrics_shape", f"{metrics.shape[0]} x {metrics.shape[1]}", metrics_source)
if val_metric is not None:
    add("validation_primary_metric", f"{primary}={val_metric:.4g}", metrics_source)
else:
    add("validation_primary_metric", f"{primary}; columns={', '.join(map(str, metrics.columns))}", metrics_source)
add("baseline_model_type", model_type_str(baseline), baseline_source)
add("baseline_hyperparameters", hp_str(baseline), baseline_source)
add("alternative_model_type", model_type_str(alt), model_source)
add("alternative_hyperparameters", hp_str(alt), model_source)
add("top_feature", tf if tf else "not available from coef_/feature_importances_", tf_src)
add("limitations", "Descriptive only; no refitting, no test exposure. Validation holdout may be small. Importances/coefficients are not causal. Count outcome may benefit from Poisson/log1p loss. Hyperparameters are as frozen in the saved model.", model_source)

interp = pd.DataFrame(records).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"model={model_type_str(model)}")
print(f"validation_{primary}={val_metric if val_metric is not None else 'NA'}")
print(f"top_feature={tf if tf else 'NA'}")
print("saved interpretation.csv")
