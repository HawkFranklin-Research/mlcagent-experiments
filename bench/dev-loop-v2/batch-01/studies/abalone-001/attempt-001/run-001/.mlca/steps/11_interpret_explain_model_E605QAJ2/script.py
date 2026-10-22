import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    return json.loads(path.read_text())

protocol = load_json(inp["_protocol"])
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
selector_all = load_pkl(inp["06_features_retain_features_4S51ZGZN/selector_all.pkl"])
baseline = load_pkl(inp["07_model_train_baseline_4EB5V15N/model.pkl"])
latest = load_pkl(inp["latest_model"])
metrics_df = pd.read_csv(inp["09_validate_score_validation_77M5Y6B8/validation_metrics.csv"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

def final_estimator(obj):
    if hasattr(obj, "named_steps"):
        return final_estimator(list(obj.named_steps.values())[-1])
    if hasattr(obj, "regressor_"):
        return final_estimator(obj.regressor_)
    if hasattr(obj, "estimators_") and len(obj.estimators_):
        return final_estimator(obj.estimators_[0])
    return obj

def extract_selector(obj):
    if hasattr(obj, "named_steps"):
        for name, step in obj.named_steps.items():
            if "select" in name.lower() or hasattr(step, "get_support"):
                return step
        return None
    return obj if hasattr(obj, "get_support") else None

def selected_names(sel, cols):
    if hasattr(sel, "named_steps"):
        sel = list(sel.named_steps.values())[-1]
    if sel is None:
        return cols
    if hasattr(sel, "get_feature_names_out"):
        try:
            return list(sel.get_feature_names_out())
        except Exception:
            pass
    if hasattr(sel, "get_support"):
        try:
            support = sel.get_support()
            if hasattr(sel, "feature_names_in_"):
                return list(np.asarray(sel.feature_names_in_)[support])
            return [c for c, m in zip(cols, support) if m]
        except Exception:
            pass
    if hasattr(sel, "feature_names_in_"):
        return list(sel.feature_names_in_)
    return cols

def importance_vector(est):
    if hasattr(est, "coef_"):
        return np.ravel(np.asarray(est.coef_))
    if hasattr(est, "feature_importances_"):
        return np.ravel(np.asarray(est.feature_importances_))
    return None

def top_from_model(model, cols, fallback_selector):
    sel = extract_selector(model)
    if sel is None:
        sel = fallback_selector
    names = selected_names(sel, cols)
    est = final_estimator(model)
    imp = importance_vector(est)
    if imp is None or len(imp) == 0:
        return None
    if len(names) != len(imp):
        names = cols if len(cols) == len(imp) else [f"feature_{i}" for i in range(len(imp))]
    return str(names[int(np.argmax(np.abs(imp)))])

top_feature = top_from_model(latest, feature_cols, selector_all) or top_from_model(baseline, feature_cols, selector_all) or "unavailable"

est = final_estimator(latest)
param_items = []
if hasattr(est, "get_params"):
    for k, v in est.get_params(deep=False).items():
        if isinstance(v, (str, int, float, bool, type(None))) and len(param_items) < 12:
            param_items.append((k, v))

def sig4(x):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(float(f"{x:.4g}"))
    return str(x)

def find_metric(df, metric):
    cols = {str(c).lower(): c for c in df.columns}
    m = metric.lower()
    if m in cols:
        vals = pd.to_numeric(df[cols[m]], errors="coerce").dropna()
        if len(vals):
            return float(vals.iloc[0])
    if "metric" in cols and "value" in cols:
        sub = df[df[cols["metric"]].astype(str).str.lower() == m]
        if "split" in cols and len(sub):
            val_sub = sub[sub[cols["split"]].astype(str).str.lower().isin(["val", "valid", "validation"])]
            if len(val_sub):
                sub = val_sub
        vals = pd.to_numeric(sub[cols["value"]], errors="coerce").dropna()
        if len(vals):
            return float(vals.iloc[0])
    for c in df.columns:
        lc = str(c).lower()
        if m in lc and ("val" in lc or "valid" in lc):
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    for c in df.columns:
        if m in str(c).lower():
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    return None

metric_value = find_metric(metrics_df, primary_metric)

rows = [
    ("model", "latest_model_file", "alternative_model.pkl"),
    ("model", "preprocessor_file", "preprocessor.pkl"),
    ("model", "selector_file", "selector_all.pkl"),
    ("model", "baseline_model_file", "model.pkl"),
]
if hasattr(latest, "named_steps"):
    rows.append(("model", "pipeline_steps", " > ".join(latest.named_steps.keys())))
rows.append(("model", "final_estimator", type(est).__name__))
for k, v in param_items:
    rows.append(("hyperparameters", k, sig4(v)))
rows += [
    ("inputs", "targets", ", ".join(targets)),
    ("inputs", "n_train_rows", str(len(train))),
    ("inputs", "n_val_rows", str(len(val))),
    ("inputs", "n_features", str(len(feature_cols))),
    ("inputs", "feature_columns", ", ".join(feature_cols)),
    ("inputs", "excluded_columns", ", ".join(excluded) if excluded else "none"),
]
if group:
    rows.append(("inputs", "group_column", str(group)))
rows += [
    ("validation", "primary_metric", str(primary_metric)),
    ("validation", "metric_value", sig4(metric_value) if metric_value is not None else "unavailable"),
    ("validation", "metric_source", "09_validate_score_validation_77M5Y6B8/validation_metrics.csv"),
    ("limitations", "note", "Metrics read from the stored validation artifact; no fitting or re-scoring performed."),
    ("limitations", "note", "Validation evidence is a single split; test data were not used."),
    ("limitations", "note", "Hyperparameters and inputs come from the frozen model and protocol."),
    ("limitations", "note", "Feature ranking comes from stored coefficients/importances, not a refit."),
    ("interpretation", "top_feature", str(top_feature)),
]
interp = pd.DataFrame(rows, columns=["section", "item", "value"])
interp = interp.head(50)
interp["value"] = interp["value"].astype(str)

mlca_runtime.save_table(interp, "interpretation.csv", "Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", str(top_feature), source="interpretation.csv")

print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"n_features={len(feature_cols)}")
print(f"metric={primary_metric} value={metric_value}")
print(f"top_feature={top_feature}")
print("saved interpretation.csv")
