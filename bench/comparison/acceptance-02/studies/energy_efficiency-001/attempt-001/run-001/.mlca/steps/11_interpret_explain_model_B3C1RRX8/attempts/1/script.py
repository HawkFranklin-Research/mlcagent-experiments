import mlca_runtime, json, pickle, numpy as np, pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
keylike = [c for c in feature_cols if train[c].dtype == object and train[c].nunique() == len(train)]
feature_cols = [c for c in feature_cols if c not in keylike]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

manifests = {}
for k in ["05_preprocess_fit_preprocessing_RHXAHV44/manifest.json",
          "06_features_retain_features_W82TE9KX/manifest.json",
          "07_model_train_baseline_1GT1TPB8/manifest.json",
          "08_model_train_alternative_KVXQRP8Q/manifest.json",
          "09_validate_score_validation_Q8GGH7XB/manifest.json"]:
    if k in inputs:
        manifests[k] = json.loads(inputs[k].read_text())

model_manifest = manifests.get("08_model_train_alternative_KVXQRP8Q/manifest.json", {})
hyperparams = model_manifest.get("hyperparameters", model_manifest.get("params", {}))

val_metrics = pd.read_csv(inputs["09_validate_score_validation_Q8GGH7XB/validation_metrics.csv"])
if "rmse" in val_metrics.columns:
    if "target" in val_metrics.columns:
        rmse_vals = val_metrics.set_index("target")["rmse"]
    else:
        rmse_vals = val_metrics["rmse"]
    mean_rmse = float(rmse_vals.mean())
else:
    rmse_cols = [c for c in val_metrics.columns if c.startswith("rmse")]
    mean_rmse = float(val_metrics[rmse_cols].mean().mean()) if rmse_cols else np.nan
    rmse_vals = None

def get_top_feature(model, feature_cols):
    steps = list(model.named_steps.items()) if hasattr(model, 'named_steps') else []
    est = steps[-1][1] if steps else model
    def get_imp(e):
        if hasattr(e, 'feature_importances_'):
            return e.feature_importances_
        if hasattr(e, 'coef_'):
            c = np.abs(e.coef_)
            if c.ndim > 1:
                c = c.mean(axis=0)
            return c
        return None
    imp = get_imp(est)
    if imp is None and hasattr(est, 'estimators_'):
        imps = [get_imp(e) for e in est.estimators_]
        imps = [i for i in imps if i is not None]
        if imps:
            imp = np.mean(imps, axis=0)
    if imp is None:
        return None
    names = None
    for name, step in reversed(steps[:-1]):
        if hasattr(step, 'get_feature_names_out'):
            try:
                names = step.get_feature_names_out()
                break
            except Exception:
                pass
    if names is None:
        for name, step in steps:
            if hasattr(step, 'feature_names_in_'):
                names = step.feature_names_in_
                break
        if names is None:
            names = feature_cols
    idx = int(np.argmax(imp))
    return str(names[idx]) if idx < len(names) else f"feature_{idx}"

top_feature = get_top_feature(model, feature_cols)

rows = []
model_type = f"Pipeline({list(model.named_steps.keys())})" if hasattr(model, 'named_steps') else type(model).__name__
rows.append({"section": "model", "item": "estimator", "value": model_type})
rows.append({"section": "model", "item": "n_features_in", "value": len(feature_cols)})
rows.append({"section": "model", "item": "targets", "value": ", ".join(map(str, targets))})
rows.append({"section": "model", "item": "primary_metric", "value": primary_metric})
if isinstance(hyperparams, dict):
    for k, v in hyperparams.items():
        rows.append({"section": "hyperparameters", "item": k, "value": str(v)})
rows.append({"section": "inputs", "item": "train_rows", "value": len(train)})
rows.append({"section": "inputs", "item": "val_rows", "value": len(val)})
rows.append({"section": "inputs", "item": "feature_columns", "value": len(feature_cols)})
rows.append({"section": "inputs", "item": "keylike_excluded", "value": ", ".join(keylike) if keylike else "none"})
rows.append({"section": "validation", "item": f"mean_{primary_metric}", "value": mean_rmse})
if rmse_vals is not None and isinstance(rmse_vals, pd.Series):
    for t, v in rmse_vals.items():
        rows.append({"section": "validation", "item": f"{primary_metric}_{t}", "value": float(v)})
rows.append({"section": "limitations", "item": "note", "value": "No fitting performed; interpretation from frozen artifacts."})
rows.append({"section": "limitations", "item": "note", "value": "Top feature from final estimator importances/coefficients."})
rows.append({"section": "limitations", "item": "note", "value": "Hyperparameters read from saved model manifest."})

interp_df = pd.DataFrame(rows)
interp_df["value"] = interp_df["value"].astype(str)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Frozen model interpretation")

if not np.isnan(mean_rmse):
    mlca_runtime.record_metric(f"mean_{primary_metric}", mean_rmse, split="val")
if top_feature:
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
print(f"mean rmse={mean_rmse}")
print(f"top_feature={top_feature}")
