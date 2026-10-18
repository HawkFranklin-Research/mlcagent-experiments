import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
req = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in req if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol)}")
if "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(protocol['task'])}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

def key_like(s):
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().all() and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        n = s.name.lower()
        return s.notna().all() and s.nunique(dropna=False) == len(s) and (n.endswith("id") or n in ("instant", "index", "row_id"))
    return False

feature_cols = [c for c in train.columns if c not in set(targets + excluded + [group]) and not key_like(train[c])]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

baseline = load_pkl(inp["07_model_train_baseline_V4Y4XVKZ/model.pkl"])
alt = load_pkl(inp["08_model_train_alternative_DHX507BF/alternative_model.pkl"])
selector = load_pkl(inp["06_features_retain_features_5J0R0BF8/selector_all.pkl"])

def unwrap(o):
    if isinstance(o, dict):
        for k in ("model", "estimator", "pipeline", "best_estimator_"):
            if k in o:
                return unwrap(o[k])
    return o

def final_est(o):
    o = unwrap(o)
    if hasattr(o, "named_steps") and o.named_steps:
        return final_est(list(o.named_steps.values())[-1])
    if hasattr(o, "steps") and o.steps:
        return final_est(o.steps[-1][1])
    for a in ("regressor_", "estimator_", "model_"):
        if hasattr(o, a):
            return final_est(getattr(o, a))
    return o

def feat_names(o, sel=None):
    o = unwrap(o)
    if hasattr(o, "get_support") and hasattr(o, "feature_names_in_"):
        try:
            sup = np.asarray(o.get_support())
            fn = list(o.feature_names_in_)
            if len(fn) == len(sup):
                return [f for f, s in zip(fn, sup) if s]
        except Exception:
            pass
    if hasattr(o, "feature_names_in_"):
        return list(o.feature_names_in_)
    if hasattr(o, "named_steps"):
        for step in reversed(list(o.named_steps.values())):
            fn = feat_names(step, sel)
            if fn is not None:
                return fn
    if sel is not None and hasattr(sel, "get_support") and hasattr(sel, "feature_names_in_"):
        try:
            sup = np.asarray(sel.get_support())
            fn = list(sel.feature_names_in_)
            if len(fn) == len(sup):
                return [f for f, s in zip(fn, sup) if s]
        except Exception:
            pass
    return None

def importance(o):
    est = final_est(o)
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        return np.abs(c if c.ndim == 1 else np.mean(np.abs(c), axis=0))
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    return None

def top_from(o):
    imp = importance(o)
    names = feat_names(o, selector)
    if imp is None:
        return None
    if names is None or len(names) != len(imp):
        names = [f"feature_{i}" for i in range(len(imp))]
    return str(names[int(np.argmax(imp))])

top_feature = top_from(baseline) or top_from(alt) or "unknown"

def params_rows(o):
    est = final_est(o)
    params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
    keys = ["n_estimators", "max_depth", "learning_rate", "min_samples_leaf", "min_samples_split",
            "subsample", "max_features", "loss", "alpha", "l1_ratio", "C", "epsilon", "gamma",
            "kernel", "solver", "penalty", "max_iter", "l2_regularization", "max_leaf_nodes",
            "min_child_samples"]
    out = [(k, params[k]) for k in keys if k in params and params[k] is not None]
    if not out:
        out = [(k, v) for k, v in params.items() if isinstance(v, (str, int, float, bool)) and k != "random_state"][:8]
    return out[:12]

def fmt(x):
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return f"{float(x):.4g}"
    return str(x)

vm = pd.read_csv(inp["09_validate_score_validation_T0YRBVEV/validation_metrics.csv"])
def metric_pairs(df):
    low = {c.lower(): c for c in df.columns}
    if "metric" in low and "value" in low:
        return [(str(r[low["metric"]]), r[low["value"]]) for _, r in df.iterrows()]
    out = []
    if len(df):
        r = df.iloc[0]
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append((c, r[c]))
    return out

metrics = metric_pairs(vm)
rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": str(item), "value": fmt(value), "source": str(source)})

for label, obj, src in [
    ("baseline", baseline, "07_model_train_baseline_V4Y4XVKZ/model.pkl"),
    ("alternative", alt, "08_model_train_alternative_DHX507BF/alternative_model.pkl"),
]:
    o = unwrap(obj)
    add("model", f"{label}_family", type(final_est(o)).__name__, src)
    steps = " > ".join(list(o.named_steps)) if hasattr(o, "named_steps") else "single estimator"
    add("model", f"{label}_pipeline_steps", steps, src)

add("input", "n_train_rows", len(train), "train.parquet")
add("input", "n_val_rows", len(val), "val.parquet")
add("input", "n_features", len(feature_cols), "train.parquet")
add("input", "target", ",".join(targets), "_protocol")
add("input", "primary_metric", protocol["task"]["primary_metric"], "_protocol")

for k, v in params_rows(baseline):
    add("hyperparameter", f"baseline_{k}", v, "07_model_train_baseline_V4Y4XVKZ/model.pkl")

for m, v in metrics[:8]:
    add("validation", m, v, "09_validate_score_validation_T0YRBVEV/validation_metrics.csv")

add("interpretation", "top_feature", top_feature, "07_model_train_baseline_V4Y4XVKZ/model.pkl")

for k, v in [
    ("scope", "Frozen fitted model; no refitting or test exposure in this step."),
    ("time", "Single chronological hourly series; validation evidence is not a future test."),
    ("features", "Excluded leakage/index columns: " + ", ".join(excluded) + "."),
    ("importance", "Importance is model-specific association, not causal effect."),
    ("metrics", "Only declared train/validation partitions are available."),
]:
    add("limitation", k, v, "interpretation.csv")

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation and validation evidence.")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train={train.shape}, val={val.shape}, features={len(feature_cols)}")
print(f"primary_metric={protocol['task']['primary_metric']}")
print(f"top_feature={top_feature}")
print(f"saved interpretation.csv rows={len(df)}")
