import mlca_runtime, pickle, json, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
outdir = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

model = load_pkl("07_model_train_baseline_16VA3CWB/model.pkl")
alt_model = load_pkl("08_model_train_alternative_YMQ80SAJ/alternative_model.pkl")
preprocessor = load_pkl("05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl")
selector = load_pkl("06_features_retain_features_W9C3S8F0/selector_all.pkl")

def get_final_estimator(m):
    if isinstance(m, Pipeline):
        return m.steps[-1][1]
    return m

def get_feature_names(sel, prep):
    if sel is None:
        return None
    try:
        names = sel.get_feature_names_out()
        if names is not None:
            return list(names)
    except Exception:
        pass
    if hasattr(sel, 'get_support') and prep is not None:
        try:
            support = sel.get_support()
            base = prep.get_feature_names_out()
            return [n for n, s in zip(base, support) if s]
        except Exception:
            pass
    return None

feat_names = get_feature_names(selector, preprocessor)
if feat_names is None and isinstance(model, Pipeline):
    for step in model.named_steps.values():
        if hasattr(step, 'get_feature_names_out'):
            try:
                feat_names = list(step.get_feature_names_out())
                break
            except Exception:
                pass

def get_inputs(m):
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)
    if isinstance(m, Pipeline):
        for step in m.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

raw_inputs = get_inputs(model)
if raw_inputs is None and isinstance(model, Pipeline):
    raw_inputs = get_inputs(model.steps[0][1])

est = get_final_estimator(model)
top_feature = "unknown"
if hasattr(est, 'coef_'):
    coef = est.coef_
    if coef.ndim > 1:
        coef = coef[0]
    idx = int(np.argmax(np.abs(coef)))
    top_feature = feat_names[idx] if feat_names is not None and idx < len(feat_names) else f"feature_{idx}"
elif hasattr(est, 'feature_importances_'):
    imp = est.feature_importances_
    idx = int(np.argmax(imp))
    top_feature = feat_names[idx] if feat_names is not None and idx < len(feat_names) else f"feature_{idx}"

val_metrics = pd.read_csv(inputs["09_validate_score_validation_DX3NN439/validation_metrics.csv"])
if 'metric' in val_metrics.columns and 'split' in val_metrics.columns:
    metric_row = val_metrics[(val_metrics['metric'] == primary_metric) & (val_metrics['split'] == 'val')]
else:
    metric_row = val_metrics.head(1)
val_value = np.nan
if len(metric_row) > 0:
    if 'value' in metric_row.columns:
        val_value = metric_row['value'].iloc[0]
    else:
        num_cols = metric_row.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            val_value = metric_row[num_cols[0]].iloc[0]

def hyper_summary(m):
    est = get_final_estimator(m)
    if hasattr(est, 'get_params'):
        params = est.get_params(deep=False)
        keep = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'alpha', 'hidden_layer_sizes', 'n_neighbors', 'max_iter']
        sel = {k: v for k, v in params.items() if k in keep}
        if not sel:
            sel = {k: v for k, v in list(params.items())[:4]}
        return json.dumps(sel, default=str)
    return type(est).__name__

limitations = ("No test set evaluation; validation metrics only; model frozen; "
               "no hyperparameter tuning at interpretation; feature importance may be biased for correlated features.")

rows = [
    ("model_type", type(model).__name__),
    ("estimator", type(get_final_estimator(model)).__name__),
    ("hyperparameters", hyper_summary(model)),
    ("raw_inputs", ", ".join(raw_inputs) if raw_inputs else "unknown"),
    ("selected_features", ", ".join(feat_names) if feat_names else "unknown"),
    ("top_feature", top_feature),
    ("validation_metric", primary_metric),
    ("validation_value", val_value),
    ("limitations", limitations),
]
interp = pd.DataFrame(rows, columns=["aspect", "value"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows {len(train)}, val rows {len(val)}")
print(f"model {type(get_final_estimator(model)).__name__} top_feature={top_feature}")
print(f"val {primary_metric}={val_value}")
print(f"selected features count={len(feat_names) if feat_names else 0}")
print("saved interpretation.csv")
