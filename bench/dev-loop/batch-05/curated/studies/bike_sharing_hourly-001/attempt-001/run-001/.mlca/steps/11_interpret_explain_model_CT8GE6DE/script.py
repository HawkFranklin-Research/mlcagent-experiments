import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
task = proto["task"]
metric = task["primary_metric"]

man_model = json.loads(inputs["07_model_train_baseline_MY9YJS0R/manifest.json"].read_text())
man_alt = json.loads(inputs["08_model_train_alternative_VYCC8X2X/manifest.json"].read_text())
man_pre = json.loads(inputs["05_preprocess_fit_preprocessing_VM7ERF98/manifest.json"].read_text())
man_sel = json.loads(inputs["06_features_retain_features_RGQ2HW1B/manifest.json"].read_text())
man_val = json.loads(inputs["09_validate_score_validation_6CVGV8P9/manifest.json"].read_text())

val_metrics = pd.read_csv(inputs["09_validate_score_validation_6CVGV8P9/validation_metrics.csv"])

with open(inputs["07_model_train_baseline_MY9YJS0R/model.pkl"], "rb") as f:
    model = pickle.load(f)
with open(inputs["08_model_train_alternative_VYCC8X2X/alternative_model.pkl"], "rb") as f:
    alt_model = pickle.load(f)
with open(inputs["06_features_retain_features_RGQ2HW1B/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def final_est(m):
    if hasattr(m, "named_steps"):
        return list(m.named_steps.values())[-1]
    return m

def feat_names(m):
    if hasattr(m, "get_feature_names_out"):
        try:
            return list(m.get_feature_names_out())
        except Exception:
            pass
    if hasattr(m, "named_steps"):
        for s in m.named_steps.values():
            if hasattr(s, "get_feature_names_out"):
                try:
                    return list(s.get_feature_names_out())
                except Exception:
                    pass
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    return None

def params_str(e):
    try:
        p = e.get_params()
        keep = {k: str(v) for k, v in p.items() if not k.startswith("__") and not hasattr(v, "fit")}
        return json.dumps(dict(list(keep.items())[:12]))
    except Exception:
        return ""

est = final_est(model)
alt_est = final_est(alt_model)
names = feat_names(selector) or feat_names(model)

imp = None
if hasattr(est, "feature_importances_"):
    imp = np.asarray(est.feature_importances_)
elif hasattr(est, "coef_"):
    c = np.asarray(est.coef_)
    imp = np.abs(c).ravel() if c.ndim > 1 else np.abs(c)

top = None
if imp is not None and names is not None and len(names) == len(imp):
    top = names[int(np.argmax(imp))]

val_score = None
try:
    for c in val_metrics.columns:
        col = val_metrics[c]
        if col.dtype.kind in ("O", "U", "S"):
            mask = col.astype(str).str.lower().str.contains(metric.lower(), na=False)
            if mask.any():
                row = val_metrics[mask].iloc[0]
                for c2 in val_metrics.columns:
                    try:
                        val_score = float(row[c2])
                        break
                    except (ValueError, TypeError):
                        continue
                break
except Exception:
    pass

rows = [
    {"item": "model_type", "value": type(est).__name__},
    {"item": "primary_metric", "value": metric},
    {"item": "targets", "value": ",".join(targets)},
    {"item": "excluded_columns", "value": ",".join(excluded)},
    {"item": "group_column", "value": str(group)},
    {"item": "n_selected_features", "value": str(len(names)) if names else "unknown"},
    {"item": "top_feature", "value": str(top)},
    {"item": "baseline_params", "value": params_str(est)},
    {"item": "alternative_model_type", "value": type(alt_est).__name__},
    {"item": "alternative_params", "value": params_str(alt_est)},
    {"item": "validation_score", "value": str(val_score) if val_score is not None else "unknown"},
    {"item": "baseline_manifest_keys", "value": ",".join(sorted(man_model.keys()))[:300]},
    {"item": "alternative_manifest_keys", "value": ",".join(sorted(man_alt.keys()))[:300]},
    {"item": "preprocess_manifest_keys", "value": ",".join(sorted(man_pre.keys()))[:300]},
    {"item": "selector_manifest_keys", "value": ",".join(sorted(man_sel.keys()))[:300]},
    {"item": "validation_manifest_keys", "value": ",".join(sorted(man_val.keys()))[:300]},
    {"item": "limitations", "value": "Frozen model; no refitting. Interpretation from saved artifacts only."},
]

interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

if top is not None:
    mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

print("rows", len(interp))
print("top_feature", top)
print("val_score", val_score)
