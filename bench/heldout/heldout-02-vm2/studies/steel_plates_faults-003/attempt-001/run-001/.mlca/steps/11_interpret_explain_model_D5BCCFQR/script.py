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
parts = [f"{k}={params[k]}" for k in sorted(params)
         if isinstance(params[k], (int, float, str, bool)) or params[k] is None]
hyper_str = "; ".join(parts[:15]) if parts else "unavailable"

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

def names_for(m, n):
    if hasattr(m, "feature_names_in_") and len(m.feature_names_in_) == n:
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for _, step in reversed(list(m.named_steps.items())):
            if hasattr(step, "get_feature_names_out"):
                try:
                    nm = list(step.get_feature_names_out())
                    if len(nm) == n:
                        return nm
                except Exception:
                    pass
            if hasattr(step, "feature_names_in_") and len(step.feature_names_in_) == n:
                return list(step.feature_names_in_)
    if hasattr(m, "get_feature_names_out"):
        try:
            nm = list(m.get_feature_names_out())
            if len(nm) == n:
                return nm
        except Exception:
            pass
    return None

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif hasattr(final, "feature_names_in_"):
    feature_names = list(final.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | {group}
    feature_names = [c for c in train.columns if c not in drop]
n_features = len(feature_names)

top_feature = "unavailable"
top_importance = "unavailable"
if imp is not None:
    nm = names_for(model, len(imp)) or (feature_names if len(feature_names) == len(imp) else None)
    if nm is not None:
        i = int(np.argmax(imp))
        top_feature = str(nm[i])
        top_importance = f"{float(imp[i]):.4g}"
    else:
        imp_note += "; names unmapped to inputs"

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
primary_val = "unavailable"
if val_metrics is not None:
    if {"metric", "value"}.issubset(val_metrics.columns):
        hit = val_metrics[val_metrics["metric"].astype(str).str.lower() == primary.lower()]
        if len(hit):
            primary_val = f"{float(hit['value'].iloc[0]):.4g}"
    elif primary in val_metrics.columns:
        primary_val = f"{float(val_metrics[primary].iloc[0]):.4g}"

limitations = ["No fitting performed; interpretation from stored attributes only."]
if imp is None:
    limitations.append("No built-in feature importance available.")
if top_feature == "unavailable":
    limitations.append("Top feature could not be mapped to an input name.")
if val_metrics is None:
    limitations.append("Validation metrics file unavailable.")
lim_str = " ".join(limitations)

rows = [
    ("model", "class", model_class),
    ("model", "n_features", str(n_features)),
    ("model", "hyperparameters", hyper_str),
    ("model", "importance_method", imp_note),
    ("model", "top_feature", top_feature),
    ("model", "top_importance", top_importance),
    ("inputs", "feature_columns", str(n_features)),
    ("inputs", "targets", ";".join(targets)),
    ("inputs", "train_rows", str(len(train))),
    ("inputs", "val_rows", str(len(val))),
    ("validation", "primary_metric", primary),
    ("validation", "primary_value", primary_val),
    ("limitations", "notes", lim_str),
]
src = "interpretation.csv"
interp = pd.DataFrame(
    [{"section": s, "item": i, "value": str(v), "source": src} for s, i, v in rows]
)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

if top_feature != "unavailable":
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"model={model_class} n_features={n_features}")
print(f"top_feature={top_feature} importance={top_importance}")
print(f"{primary}={primary_val} val_rows={len(val)}")
