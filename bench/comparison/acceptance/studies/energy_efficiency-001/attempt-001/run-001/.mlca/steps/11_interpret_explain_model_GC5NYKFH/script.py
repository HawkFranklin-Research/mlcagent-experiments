import mlca_runtime, json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
metric_name = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
with open(inp["06_features_retain_features_J9XJJ7CH/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inp["latest_model"], "rb") as f:
    latest = pickle.load(f)
val = pd.read_csv(inp["09_validate_score_validation_FBE2D3SM/validation_metrics.csv"])

def is_keylike(s):
    s2 = s.dropna()
    if len(s2) == 0:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return pd.api.types.is_integer_dtype(s) and s2.is_unique and len(s2) > 1
    return s2.nunique() == len(s2) and len(s2) > 1

skip = set(targets) | set(excluded)
if group is not None:
    skip.add(group)
features = [c for c in train.columns if c not in skip and not is_keylike(train[c])]

def final_est(model):
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model

def simple_params(est, limit=15):
    out = []
    try:
        params = est.get_params(deep=False)
    except Exception:
        return out
    for k, v in params.items():
        if isinstance(v, (str, bool, int, float, type(None))):
            out.append((k, v))
        elif isinstance(v, tuple) and all(isinstance(x, (str, bool, int, float, type(None))) for x in v):
            out.append((k, v))
    out.sort(key=lambda kv: kv[0])
    return out[:limit]

def selected_count(sel):
    if hasattr(sel, "get_support"):
        try:
            return int(sel.get_support().sum())
        except Exception:
            pass
    if hasattr(sel, "named_steps"):
        for step in reversed(list(sel.named_steps.values())):
            if hasattr(step, "get_support"):
                try:
                    return int(step.get_support().sum())
                except Exception:
                    pass
    return None

def importance(est):
    if hasattr(est, "estimators_"):
        vals = []
        for e in est.estimators_:
            if hasattr(e, "coef_"):
                vals.append(np.abs(np.asarray(e.coef_)).ravel())
            elif hasattr(e, "feature_importances_"):
                vals.append(np.asarray(e.feature_importances_).ravel())
        if vals:
            return np.mean(np.vstack(vals), axis=0)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    return None

def names_for(model, est, n):
    for obj in [est, model]:
        if hasattr(obj, "feature_names_in_"):
            names = list(obj.feature_names_in_)
            if len(names) == n:
                return names
    objs = [model] + (list(model.named_steps.values()) if hasattr(model, "named_steps") else [])
    for obj in objs:
        if hasattr(obj, "get_feature_names_out"):
            try:
                names = list(obj.get_feature_names_out())
                if len(names) == n:
                    return names
            except Exception:
                pass
    return [f"feature_{i}" for i in range(n)]

def extract_rmse(df, targets):
    vals = {}
    for t in targets:
        if t in df.columns:
            col = pd.to_numeric(df[t], errors="coerce").dropna()
            if len(col):
                vals[t] = float(col.iloc[0])
    if vals:
        return vals
    if "metric" in df.columns:
        rows = df[df["metric"].astype(str).str.lower().str.contains("rmse")]
        if "target" in rows.columns:
            for _, r in rows.iterrows():
                try:
                    key = str(r["target"])
                    vals[key] = float(r["value"]) if "value" in rows.columns else float(r[key])
                except Exception:
                    pass
        if not vals and "value" in rows.columns and len(rows):
            vals["overall"] = float(rows["value"].iloc[0])
    if not vals:
        nums = df.select_dtypes("number")
        if nums.shape[0] and nums.shape[1]:
            vals["overall"] = float(nums.iloc[0, 0])
    return vals

est = final_est(latest)
imp = importance(est)
top, top_imp = "unavailable", np.nan
if imp is not None and len(imp) > 0:
    idx = int(np.argmax(np.abs(imp)))
    names = names_for(latest, est, len(imp))
    top = str(names[idx]) if idx < len(names) else f"feature_{idx}"
    top_imp = float(imp[idx])

vals = extract_rmse(val, targets)
target_vals = [vals[t] for t in targets if t in vals]
mean_rmse = float(np.mean(target_vals)) if target_vals else vals.get("overall")

rows = []
rows.append(("model", "frozen_model_class", type(latest).__name__, "latest_model"))
rows.append(("model", "final_estimator_class", type(est).__name__, "latest_model"))
steps = " > ".join(latest.named_steps.keys()) if hasattr(latest, "named_steps") else "single estimator"
rows.append(("model", "pipeline_steps", steps, "latest_model"))
for k, v in simple_params(est):
    rows.append(("hyperparameter", k, str(v), "latest_model"))
if hasattr(est, "estimator"):
    for k, v in simple_params(est.estimator):
        rows.append(("hyperparameter", f"estimator__{k}", str(v), "latest_model"))
rows.append(("input", "n_train_rows", str(train.shape[0]), "train"))
rows.append(("input", "n_features", str(len(features)), "train"))
rows.append(("input", "targets", ",".join(targets), "_protocol"))
rows.append(("input", "excluded_columns", ",".join(excluded) if excluded else "none", "_protocol"))
rows.append(("input", "group_column", str(group) if group else "none", "_protocol"))
sc = selected_count(selector)
if sc is not None:
    rows.append(("input", "n_selected_features", str(sc), "selector_all.pkl"))
rows.append(("model", "top_feature", top, "latest_model"))
if not np.isnan(top_imp):
    rows.append(("model", "top_feature_mean_abs_importance", str(round(top_imp, 4)), "latest_model"))
rows.append(("validation", "primary_metric", metric_name, "_protocol"))
for k, v in vals.items():
    rows.append(("validation", f"rmse_{k}", str(round(v, 4)), "validation_metrics.csv"))
if mean_rmse is not None:
    rows.append(("validation", "rmse_mean", str(round(mean_rmse, 4)), "validation_metrics.csv"))
rows.append(("limitation", "no_fitting", "interpretation only; model loaded from disk", "interpretation.csv"))
rows.append(("limitation", "importance_source", "model coefficients/feature_importances; not causal", "latest_model"))
rows.append(("limitation", "validation_only", "validation evidence reported; no test exposure", "validation_metrics.csv"))
rows.append(("limitation", "multioutput_aggregation", "importance averaged across targets where applicable", "latest_model"))

interp = pd.DataFrame(rows, columns=["section", "item", "value", "source"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
if mean_rmse is not None:
    mlca_runtime.record_metric("val_rmse_mean", float(mean_rmse), split="val")

print(f"model={type(latest).__name__} final={type(est).__name__}")
print(f"train_rows={train.shape[0]} features={len(features)} selected={sc}")
print(f"top_feature={top}")
print(f"val_rmse_mean={mean_rmse}")
print("saved interpretation.csv")
