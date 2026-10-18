import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
task = protocol["task"]
primary_metric = task["primary_metric"]

val_path = inputs["09_validate_score_validation_DBTDYA32/validation_metrics.csv"]
val_df = pd.read_csv(val_path)

def metric_value(df, name):
    if {"metric", "value"}.issubset(df.columns):
        m = df[df["metric"].astype(str).str.contains(name, case=False, na=False)]
        if len(m):
            return float(m["value"].iloc[0])
    for col in df.columns:
        if str(col).lower() == name.lower() or name.lower() in str(col).lower():
            return float(df[col].dropna().iloc[0])
    nums = df.select_dtypes(include=[np.number])
    return float(nums.iloc[0, 0]) if nums.shape[1] else np.nan

val_value = metric_value(val_df, primary_metric)

model_key = "08_model_train_alternative_BG3XZYK8/alternative_model.pkl"
with open(inputs[model_key], "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict):
    for k in ("model", "estimator", "pipeline"):
        if k in model:
            model = model[k]
            break

alt_manifest = json.loads(inputs["08_model_train_alternative_BG3XZYK8/manifest.json"].read_text())

def steps_of(obj):
    if hasattr(obj, "named_steps"):
        return list(obj.named_steps.items())
    return []

steps = steps_of(model)
est = steps[-1][1] if steps else model
model_name = type(est).__name__

def input_features(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    for _, step in steps_of(obj):
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)
    return []

feats = input_features(model)

def get_top(obj):
    items = steps_of(obj)
    est = items[-1][1] if items else obj
    prev = items[-2][1] if len(items) > 1 else None
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        vals = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        return "unknown"
    names = list(getattr(est, "feature_names_in_", []))
    if prev is not None and hasattr(prev, "get_support"):
        try:
            support = prev.get_support()
            base = list(getattr(prev, "feature_names_in_", []))
            if len(support) == len(vals) and len(base) == len(support):
                names = [n for n, s in zip(base, support) if s]
        except Exception:
            pass
    if len(names) != len(vals):
        names = [f"f{i}" for i in range(len(vals))]
    return str(names[int(np.argmax(vals))])

top = get_top(model)

hyper = (
    alt_manifest.get("hyperparameters")
    or alt_manifest.get("best_params")
    or alt_manifest.get("params")
    or alt_manifest.get("model_params")
)
if hyper is None:
    hyper = {k: v for k, v in alt_manifest.items() if any(s in k.lower() for s in ("param", "hyper", "model", "estimator"))}
if not hyper:
    hyper = alt_manifest
hyper_str = json.dumps(hyper, default=str)[:1000]

rows = [
    dict(section="model", key="model_type", text_value=model_name, numeric_value=np.nan, source=model_key),
    dict(section="model", key="pipeline_steps", text_value=str([n for n, _ in steps]) if steps else "single estimator", numeric_value=np.nan, source=model_key),
    dict(section="task", key="targets", text_value=json.dumps(targets), numeric_value=np.nan, source="_protocol"),
    dict(section="task", key="primary_metric", text_value=primary_metric, numeric_value=np.nan, source="_protocol"),
    dict(section="validation", key=primary_metric, text_value=np.nan, numeric_value=val_value, source="09_validate_score_validation_DBTDYA32/validation_metrics.csv"),
    dict(section="inputs", key="n_features", text_value=np.nan, numeric_value=len(feats), source=model_key),
    dict(section="inputs", key="features", text_value=json.dumps(feats), numeric_value=np.nan, source=model_key),
    dict(section="hyperparameters", key="selected", text_value=hyper_str, numeric_value=np.nan, source="08_model_train_alternative_BG3XZYK8/manifest.json"),
    dict(section="evidence", key="validation_file", text_value="09_validate_score_validation_DBTDYA32/validation_metrics.csv", numeric_value=np.nan, source="09_validate_score_validation_DBTDYA32/validation_metrics.csv"),
    dict(section="limitations", key="note", text_value="Frozen model described without refitting; validation evidence is saved; feature names are model-declared.", numeric_value=np.nan, source="08_model_train_alternative_BG3XZYK8/manifest.json"),
    dict(section="fact", key="top_feature", text_value=top, numeric_value=np.nan, source=model_key),
]
df = pd.DataFrame(rows, columns=["section", "key", "text_value", "numeric_value", "source"])

def round_sig(x, sig=4):
    if isinstance(x, (int, float, np.number)) and not pd.isna(x):
        return 0 if x == 0 else float(f"{x:.{sig}g}")
    return x

df["numeric_value"] = df["numeric_value"].apply(round_sig)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")

mlca_runtime.record_fact_candidate("top_feature", top, source=model_key)
if pd.notna(val_value):
    mlca_runtime.record_metric(primary_metric, float(val_value), "val")

print(f"rows={len(df)} model={model_name} features={len(feats)} top={top} {primary_metric}={val_value}")
