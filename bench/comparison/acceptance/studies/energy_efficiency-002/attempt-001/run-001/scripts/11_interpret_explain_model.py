import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

def read_json(p):
    return json.loads(p.read_text())

def read_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

protocol = read_json(inputs["_protocol"])
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")
if "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(protocol['task'].keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
base_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | {group}]
feature_cols = []
for c in base_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("string") or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and s.is_monotonic_increasing:
        continue
    feature_cols.append(c)

model = read_pickle(inputs["latest_model"])
preprocessor = read_pickle(inputs["05_preprocess_fit_preprocessing_9HEDMWQR/preprocessor.pkl"])
selector = read_pickle(inputs["06_features_retain_features_HD824TRJ/selector_all.pkl"])

def short_obj(x, limit=700):
    s = repr(x)
    return s if len(s) <= limit else s[:limit] + "..."

def model_summary(m):
    if hasattr(m, "named_steps"):
        steps = " > ".join(f"{k}:{type(v).__name__}" for k, v in m.named_steps.items())
        final = list(m.named_steps.values())[-1]
        params = final.get_params(deep=False) if hasattr(final, "get_params") else {}
        return f"Pipeline[{steps}]; final={type(final).__name__}; params={short_obj(params)}"
    params = m.get_params(deep=False) if hasattr(m, "get_params") else {}
    return f"{type(m).__name__}; params={short_obj(params)}"

def output_names(m):
    if m is None:
        return None
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            n = output_names(step)
            if n is not None:
                return n
    if hasattr(m, "feature_names_in_"):
        try:
            return list(m.feature_names_in_)
        except Exception:
            pass
    if hasattr(m, "get_feature_names_out"):
        try:
            return list(m.get_feature_names_out())
        except Exception:
            pass
    return None

def single_importance(est):
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        return np.mean(np.abs(c), axis=0) if c.ndim > 1 else np.abs(c)
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_).ravel()
    return None

def extract_importance(m):
    obj = m
    if hasattr(obj, "named_steps"):
        for step in reversed(list(obj.named_steps.values())):
            if any(hasattr(step, a) for a in ("coef_", "feature_importances_", "estimators_")):
                obj = step
                break
    vals = []
    if hasattr(obj, "estimators_"):
        for est in obj.estimators_:
            v = single_importance(est)
            if v is not None:
                vals.append(v)
        if vals:
            return np.mean(np.vstack(vals), axis=0)
    return single_importance(obj)

imp = extract_importance(model)
names = output_names(model)
if names is None or imp is None or len(names) != len(imp):
    names = output_names(selector)
if names is None or imp is None or len(names) != len(imp):
    names = output_names(preprocessor)
if imp is not None and names is not None and len(names) == len(imp):
    top_feature = str(names[int(np.argmax(imp))])
elif imp is not None:
    top_feature = f"feature_index_{int(np.argmax(imp))}"
else:
    top_feature = "unavailable"

def validation_text(path):
    v = pd.read_csv(path)
    if v.empty:
        return "no validation metric rows"
    recs = []
    for _, r in v.head(12).iterrows():
        recs.append(", ".join(f"{c}={r[c]}" for c in v.columns if pd.notna(r[c])))
    s = " | ".join(recs)
    return s if len(s) <= 1500 else s[:1500] + "..."

rows = []
def add(aspect, detail, source):
    rows.append({"aspect": aspect, "detail": str(detail), "source": str(source)})

add("event", "Interpret frozen model without fitting", "_protocol")
add("selected_model_file", inputs["latest_model"].name, inputs["latest_model"].name)
add("model_structure", model_summary(model), inputs["latest_model"].name)
add("targets", ", ".join(targets), "_protocol")
add("primary_metric", primary_metric, "_protocol")
add("group_column", group, "_protocol")
add("excluded_columns", excluded, "_protocol")
add("feature_columns_count", len(feature_cols), inputs["train"].name)
add("feature_columns", ", ".join(feature_cols), inputs["train"].name)
add("validation_evidence", validation_text(inputs["09_validate_score_validation_AE5E8TGS/validation_metrics.csv"]), inputs["09_validate_score_validation_AE5E8TGS/validation_metrics.csv"].name)
add("top_feature", top_feature, inputs["latest_model"].name)
add("limitation", "No temporal or identifier column; the model describes simulated factorial associations, not causal effects.", "_protocol")
add("limitation", "Validation evidence only; no post-test fitting or threshold tuning was performed here.", inputs["09_validate_score_validation_AE5E8TGS/validation_metrics.csv"].name)
add("limitation", "Feature importance is model-derived and may reflect correlated design parameters.", inputs["latest_model"].name)

interp = pd.DataFrame(rows, columns=["aspect", "detail", "source"]).head(50)
saved = mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source=saved.name)

print(f"saved interpretation.csv rows={len(interp)}")
print(f"top_feature={top_feature}")
print(f"targets={targets}")
