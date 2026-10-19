import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"Missing {k}; available top-level keys: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
metric = task.get("primary_metric", "rmse")
group = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s) and len(s) > 1:
        continue
    feature_cols.append(c)

def final_estimator(obj):
    if hasattr(obj, "named_steps"):
        steps = list(obj.named_steps.values())
        if steps:
            return final_estimator(steps[-1])
    return obj

with inp["latest_model"].open("rb") as f:
    model = pickle.load(f)
est = final_estimator(model)
try:
    params = est.get_params(deep=False)
except Exception:
    params = {}
params_clean = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in params.items()}
params_json = json.dumps(params_clean, sort_keys=True)

imp = None
if hasattr(est, "feature_importances_"):
    imp = np.asarray(est.feature_importances_).ravel()
elif hasattr(est, "coef_"):
    coef = np.asarray(est.coef_)
    imp = np.abs(coef).mean(axis=0).ravel() if coef.ndim > 1 else np.abs(coef).ravel()

def names_from_model(model, feature_cols):
    if not hasattr(model, "named_steps"):
        return None
    steps = model.named_steps
    names = None
    for step in steps.values():
        if hasattr(step, "get_feature_names_out"):
            try:
                names = list(step.get_feature_names_out(feature_cols))
            except Exception:
                try:
                    names = list(step.get_feature_names_out())
                except Exception:
                    names = None
            if names is not None:
                break
    if names is None:
        names = list(feature_cols)
    for step in steps.values():
        if hasattr(step, "get_support"):
            try:
                support = step.get_support()
                if len(support) == len(names):
                    names = [n for n, s in zip(names, support) if s]
            except Exception:
                pass
    return names

top_feature = "unknown"
if imp is not None and len(imp) > 0:
    idx = int(np.argmax(imp))
    names = names_from_model(model, feature_cols)
    if names is not None and len(names) == len(imp):
        top_feature = str(names[idx])
    elif len(feature_cols) == len(imp):
        top_feature = str(feature_cols[idx])

vm = pd.read_csv(inp["09_validate_score_validation_D3AGSKPV/validation_metrics.csv"])
rmse = None
for col in vm.columns:
    if "rmse" in col.lower() and pd.api.types.is_numeric_dtype(vm[col]):
        rmse = float(vm[col].dropna().iloc[0])
        break
if rmse is None:
    for col in vm.columns:
        if pd.api.types.is_object_dtype(vm[col]) or pd.api.types.is_string_dtype(vm[col]):
            mask = vm[col].astype(str).str.lower().str.contains("rmse", na=False)
            if mask.any():
                row = vm[mask].iloc[0]
                for c2 in vm.columns:
                    if pd.api.types.is_numeric_dtype(vm[c2]) and not pd.isna(row[c2]):
                        rmse = float(row[c2])
                        break
                break

rows = []
def add(aspect, value, evidence):
    rows.append({"aspect": str(aspect), "value": str(value), "evidence_file": str(evidence)})

add("task", f"{task.get('type', 'regression')} ({metric})", "_protocol")
add("targets", ", ".join(map(str, targets)), "_protocol")
add("n_train_rows", len(train), "train")
add("n_val_rows", len(val), "val")
add("feature_count", len(feature_cols), "train")
add("feature_columns", ", ".join(map(str, feature_cols)), "train")
add("model_class", est.__class__.__name__, "latest_model")
add("selected_hyperparameters", params_json, "latest_model")
if rmse is not None:
    add("validation_rmse", float(f"{rmse:.4g}"), "09_validate_score_validation_D3AGSKPV/validation_metrics.csv")
add("top_feature", top_feature, "06_features_retain_features_Q9C0R6MS/selector_all.pkl")
add("limitations", "Frozen model described without refitting; validation-only evidence; Rings is an indirect age proxy and morphometric measurements are noisy; no external/test evaluation reported.", "latest_model")

interp = pd.DataFrame(rows, columns=["aspect", "value", "evidence_file"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
if rmse is not None:
    mlca_runtime.record_metric("validation_rmse", rmse, "val")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"rows={len(interp)} features={len(feature_cols)} top={top_feature} rmse={rmse}")
