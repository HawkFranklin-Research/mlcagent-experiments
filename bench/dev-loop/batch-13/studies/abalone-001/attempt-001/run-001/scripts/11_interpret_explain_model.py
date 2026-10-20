import mlca_runtime, pickle, json, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(k):
    with open(inputs[k], "rb") as f:
        return pickle.load(f)

model = load_pkl("07_model_train_baseline_16VA3CWB/model.pkl")

def final_est(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

est = final_est(model)

def feat_names_for(m):
    if isinstance(m, Pipeline):
        try:
            return list(m[:-1].get_feature_names_out())
        except Exception:
            pass
    steps = m.named_steps.values() if isinstance(m, Pipeline) else [m]
    for s in steps:
        if hasattr(s, "get_feature_names_out"):
            try:
                return list(s.get_feature_names_out())
            except Exception:
                continue
    return None

feat_names = feat_names_for(model)

def raw_inputs_for(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if isinstance(m, Pipeline):
        for s in m.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

raw_inputs = raw_inputs_for(model)

top_feature = "unknown"
if hasattr(est, "coef_"):
    c = np.ravel(np.asarray(est.coef_))
    idx = int(np.argmax(np.abs(c)))
    top_feature = feat_names[idx] if feat_names and idx < len(feat_names) else f"feature_{idx}"
elif hasattr(est, "feature_importances_"):
    imp = np.ravel(np.asarray(est.feature_importances_))
    idx = int(np.argmax(imp))
    top_feature = feat_names[idx] if feat_names and idx < len(feat_names) else f"feature_{idx}"

vm = pd.read_csv(inputs["09_validate_score_validation_DX3NN439/validation_metrics.csv"])
val_value = float("nan")
if "metric" in vm.columns:
    mask = vm["metric"] == primary_metric
    if "split" in vm.columns:
        mask = mask & (vm["split"] == "val")
    row = vm[mask]
    if len(row) and "value" in row.columns:
        val_value = float(row["value"].iloc[0])

def hyper_summary(m):
    e = final_est(m)
    if hasattr(e, "get_params"):
        p = e.get_params(deep=False)
        keep = ["n_estimators","max_depth","learning_rate","C","alpha",
                "hidden_layer_sizes","n_neighbors","max_iter",
                "min_samples_leaf","subsample","l1_ratio"]
        s = {k: v for k, v in p.items() if k in keep}
        if not s:
            s = dict(list(p.items())[:4])
        return json.dumps(s, default=str)
    return type(e).__name__

lim = ("Frozen model, no refit at interpretation; validation-only evidence; "
       "no test-set scoring performed; feature importance may be unstable "
       "for correlated predictors.")

def fmt(v):
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)

rows = [
    ("model_class", type(model).__name__),
    ("estimator", type(est).__name__),
    ("hyperparameters", hyper_summary(model)),
    ("raw_inputs", ", ".join(raw_inputs) if raw_inputs else "unknown"),
    ("selected_features", ", ".join(feat_names) if feat_names else "unknown"),
    ("n_selected_features", str(len(feat_names) if feat_names else 0)),
    ("top_feature", top_feature),
    ("validation_metric", primary_metric),
    ("validation_value", fmt(val_value)),
    ("train_rows", str(len(train))),
    ("val_rows", str(len(val))),
    ("limitations", lim),
]
interp = pd.DataFrame(rows, columns=["aspect", "value"])
interp["aspect"] = interp["aspect"].astype(str)
interp["value"] = interp["value"].astype(str)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train={len(train)} val={len(val)}")
print(f"model={type(est).__name__} top_feature={top_feature}")
print(f"val_{primary_metric}={fmt(val_value)}")
print(f"n_features={len(feat_names) if feat_names else 0}")
print("saved interpretation.csv")
