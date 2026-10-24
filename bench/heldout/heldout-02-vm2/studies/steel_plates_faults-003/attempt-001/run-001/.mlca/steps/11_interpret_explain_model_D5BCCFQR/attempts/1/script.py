import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print(f"train_rows={len(train)} val_rows={len(val)}")

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

def final_estimator(m):
    if hasattr(m, "named_steps"):
        return list(m.named_steps.values())[-1]
    return m

final = final_estimator(model)
model_class = type(model).__name__
if type(model) is not type(final):
    model_class += f" / {type(final).__name__}"

params = final.get_params(deep=False) if hasattr(final, "get_params") else {}
param_parts = []
for k in sorted(params):
    v = params[k]
    if isinstance(v, (int, float, str, bool, type(None))):
        param_parts.append(f"{k}={v}")
hyper_str = "; ".join(param_parts[:15])

if hasattr(final, "feature_importances_"):
    imp = np.asarray(final.feature_importances_).ravel()
    imp_note = "feature_importances_"
elif hasattr(final, "coef_"):
    coef = np.asarray(final.coef_)
    if coef.ndim > 1:
        coef = np.abs(coef).mean(axis=0)
    imp = np.abs(coef).ravel()
    imp_note = "abs(coef_)"
else:
    imp = None
    imp_note = "unavailable"

def find_names_for_importance(m, n):
    if hasattr(m, "feature_names_in_") and len(m.feature_names_in_) == n:
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for _, step in reversed(list(m.named_steps.items())):
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out())
                    if len(names) == n:
                        return names
                except Exception:
                    pass
            if hasattr(step, "feature_names_in_") and len(step.feature_names_in_) == n:
                return list(step.feature_names_in_)
    if hasattr(m, "get_feature_names_out"):
        try:
            names = list(m.get_feature_names_out())
            if len(names) == n:
                return names
        except Exception:
            pass
    return None

feature_names = []
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif hasattr(final, "feature_names_in_"):
    feature_names = list(final.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | {group}
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        feature_names.append(c)

n_features = len(feature_names)
top_feature = np.nan
top_importance = np.nan
top_names = None

if imp is not None:
    top_names = find_names_for_importance(model, len(imp))
    if top_names is None and len(feature_names) == len(imp):
        top_names = feature_names
    if top_names is not None:
        idx = int(np.argmax(imp))
        top_feature = str(top_names[idx])
        top_importance = float(imp[idx])
    else:
        imp_note += "; names not mapped to original inputs"

val_metrics = None
for key in ["09_validate_score_validation_9862PPB5/validation_metrics.csv",
            "09_validate_score_validation_9862PPB5/validation_metrics.parquet"]:
    try:
        p = inp[key]
        val_metrics = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
        break
    except Exception:
        continue

primary = task.get("primary_metric", "macro_f1")
primary_val = np.nan
if val_metrics is not None:
    if {"metric", "value"}.issubset(val_metrics.columns):
        hit = val_metrics[val_metrics["metric"].astype(str).str.lower() == primary.lower()]
        if len(hit):
            primary_val = float(hit["value"].iloc[0])
    elif primary in val_metrics.columns:
        primary_val = float(val_metrics[primary].iloc[0])
    else:
        for c in val_metrics.columns:
            if c.lower() == primary.lower():
                primary_val = float(val_metrics[c].iloc[0])
                break

limitations = "No fitting performed; interpretation uses stored attributes only."
if imp is None:
    limitations += " No built-in feature importance available."
if np.isnan(top_feature):
    limitations += " Top feature could not be mapped to original input name."
if val_metrics is None:
    limitations += " Validation metrics file unavailable."

def sig4(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x

rows = []
def add(section, item, value, source):
    rows.append({"section": section, "item": item, "value": value, "source": source})

add("model", "class", model_class, "latest_model")
add("model", "n_features", n_features, "latest_model")
add("model", "hyperparameters", hyper_str, "latest_model")
add("model", "importance_method", imp_note, "latest_model")
add("model", "top_feature", top_feature if not np.isnan(top_feature) else "unavailable", "latest_model")
add("model", "top_importance", sig4(top_importance) if not np.isnan(top_importance) else "unavailable", "latest_model")
add("validation", "primary_metric", primary, "09_validate_score_validation_9862PPB5/validation_metrics.csv")
add("validation", "primary_value", sig4(primary_val) if not np.isnan(primary_val) else "unavailable", "09_validate_score_validation_9862PPB5/validation_metrics.csv")
add("validation", "val_rows", len(val), "val")
add("limitations", "notes", limitations, "interpretation.csv")

interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

if not np.isnan(top_feature):
    mlca_runtime.record_fact_candidate("top_feature", str(top_feature), source="interpretation.csv")

print(f"model={model_class} n_features={n_features}")
print(f"top_feature={top_feature} importance={top_importance}")
print(f"{primary}={primary_val} val_rows={len(val)}")
print(f"saved interpretation.csv rows={len(interp)}")
