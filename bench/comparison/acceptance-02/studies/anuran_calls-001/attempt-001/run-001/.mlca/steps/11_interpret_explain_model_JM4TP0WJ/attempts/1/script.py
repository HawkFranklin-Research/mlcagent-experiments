import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"Missing _protocol key '{k}'. Available top-level keys: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto["group_column"]
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task key 'primary_metric'. Available task keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(s):
    if s.nunique(dropna=False) == len(s) and (pd.api.types.is_integer_dtype(s) or s.dtype == object):
        return True
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

preproc = load_pkl(inputs["05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl"])
selector = load_pkl(inputs["06_features_retain_features_J1HGBJXK/selector_all.pkl"])
model = load_pkl(inputs["07_model_train_baseline_R0Q02H91/model.pkl"])
alt_model = load_pkl(inputs["08_model_train_alternative_025XEA6J/alternative_model.pkl"])
latest = load_pkl(inputs["latest_model"])

def pipeline_steps(est):
    return list(est.named_steps.keys()) if hasattr(est, "named_steps") else []

def get_feature_names(est):
    if hasattr(est, "get_feature_names_out"):
        try: return list(est.get_feature_names_out())
        except Exception: pass
    if hasattr(est, "feature_names_in_"): return list(est.feature_names_in_)
    if hasattr(est, "named_steps"):
        for step in reversed(list(est.named_steps.values())):
            names = get_feature_names(step)
            if names is not None: return names
    return None

def get_top_feature(est):
    if est is None: return None, None
    if hasattr(est, "named_steps"):
        est = list(est.named_steps.values())[-1]
    estimators = getattr(est, "estimators_", [est])
    importances = []
    for e in estimators:
        if hasattr(e, "feature_importances_"):
            importances.append(np.abs(e.feature_importances_))
        elif hasattr(e, "coef_"):
            coef = e.coef_
            importances.append(np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef))
    if not importances: return None, None
    mean_imp = np.mean(importances, axis=0)
    idx = int(np.argmax(mean_imp))
    names = get_feature_names(est) or get_feature_names(selector) or get_feature_names(preproc)
    name = names[idx] if names is not None and idx < len(names) else f"feature_{idx}"
    return name, float(mean_imp[idx])

top_feature, top_importance = get_top_feature(latest)

val_metrics = pd.read_csv(inputs["09_validate_score_validation_Z1N606V7/validation_metrics.csv"])
metric_val = None
metric_col = next((c for c in ["metric","name","measure"] if c in val_metrics.columns), None)
value_col = next((c for c in ["value","score","metric_value"] if c in val_metrics.columns), None)
if metric_col and value_col:
    sel = val_metrics[val_metrics[metric_col].astype(str).str.contains(primary_metric, case=False, na=False)]
    if len(sel) > 0:
        metric_val = float(sel[value_col].iloc[0])
if metric_val is None:
    num_cols = val_metrics.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        metric_val = float(val_metrics[num_cols[0]].iloc[0])

if metric_val is not None:
    mlca_runtime.record_metric(primary_metric, metric_val, "val")

rows = []
def add(aspect, detail, text_value=None, numeric_value=None, source=""):
    rows.append({"aspect": aspect, "detail": detail, "text_value": text_value, "numeric_value": numeric_value, "source": source})

add("Model", "baseline_pipeline_steps", text_value=str(pipeline_steps(model)), source="07_model_train_baseline_R0Q02H91/model.pkl")
add("Model", "alternative_pipeline_steps", text_value=str(pipeline_steps(alt_model)), source="08_model_train_alternative_025XEA6J/alternative_model.pkl")
add("Model", "preprocessor_steps", text_value=str(pipeline_steps(preproc)), source="05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl")
add("Model", "selector_steps", text_value=str(pipeline_steps(selector)), source="06_features_retain_features_J1HGBJXK/selector_all.pkl")
add("Inputs", "n_train_rows", numeric_value=len(train), source="train")
add("Inputs", "n_val_rows", numeric_value=len(val), source="val")
add("Inputs", "n_features", numeric_value=len(feature_cols), source="train")
add("Inputs", "targets", text_value=str(targets), source="_protocol")
add("Inputs", "group_column", text_value=group_col, source="_protocol")
add("Inputs", "excluded_columns", text_value=str(excluded), source="_protocol")
add("Validation", primary_metric, numeric_value=metric_val if metric_val is not None else np.nan, source="09_validate_score_validation_Z1N606V7/validation_metrics.csv")
add("Validation", "metric_reason", text_value=task.get("metric_reason", ""), source="_protocol")
add("Limitations", "no_fitting_during_interpretation", text_value="true", source="script")
if top_feature:
    add("Top feature", "name", text_value=top_feature, source="interpretation.csv")
    add("Top feature", "importance", numeric_value=round(top_importance, 4), source="interpretation.csv")
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

interp_df = pd.DataFrame(rows).head(50)
for c in interp_df.columns:
    if interp_df[c].dtype == float:
        interp_df[c] = interp_df[c].round(4)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Frozen model interpretation")

print(f"Rows: train={len(train)}, val={len(val)}")
print(f"Features: {len(feature_cols)}")
print(f"Targets: {targets}")
print(f"Primary metric ({primary_metric}): {metric_val}")
if top_feature: print(f"Top feature: {top_feature}")
