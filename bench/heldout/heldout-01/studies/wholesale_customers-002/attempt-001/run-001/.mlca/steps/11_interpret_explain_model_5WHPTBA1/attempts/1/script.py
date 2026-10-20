import mlca_runtime, pickle, json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Protocol missing {key}; available keys: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
primary = task["primary_metric"]
group = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s):
    name = str(s.name).lower()
    if s.isna().any() or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return "id" in name or "code" in name or "key" in name or s.is_monotonic_increasing
    return False

exclude_set = set(targets + excluded + ([group] if group else []))
feature_cols = [c for c in train.columns if c not in exclude_set and not key_like(train[c])]

def final_est(obj):
    if hasattr(obj, "named_steps") and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    return obj

def raw_features(obj):
    if hasattr(obj, "named_steps") and obj.named_steps:
        first = list(obj.named_steps.values())[0]
        if hasattr(first, "feature_names_in_"):
            return [str(x) for x in first.feature_names_in_]
    if hasattr(obj, "feature_names_in_"):
        return [str(x) for x in obj.feature_names_in_]
    est = final_est(obj)
    if hasattr(est, "feature_names_in_"):
        return [str(x) for x in est.feature_names_in_]
    return []

model_key = "latest_model" if "latest_model" in inp else "07_model_train_baseline_9CYH36JK/model.pkl"
with open(inp[model_key], "rb") as f:
    model = pickle.load(f)
est = final_est(model)
model_name = type(est).__name__
params = {k: v for k, v in est.get_params(deep=False).items() if not hasattr(v, "fit") and not callable(v)}
param_text = "; ".join(f"{k}={v:.4g}" if isinstance(v, (float, np.floating)) else f"{k}={v}" for k, v in params.items())
if len(param_text) > 500:
    param_text = param_text[:497] + "..."
feats = raw_features(model) or feature_cols
feat_text = ", ".join(feats)
if len(feat_text) > 500:
    feat_text = feat_text[:497] + "..."

metrics_df = pd.read_csv(inp["09_validate_score_validation_FBYY3QH9/validation_metrics.csv"])
def get_metric(df, name):
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[0])
    low = {str(c).lower(): c for c in df.columns}
    if "metric" in low and "value" in low:
        hit = df[df[low["metric"]].astype(str).str.lower() == name.lower()]
        if len(hit):
            s = pd.to_numeric(hit[low["value"]], errors="coerce").dropna()
            if len(s):
                return float(s.iloc[0])
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            m = df[c].astype(str).str.lower()
            if (m == name.lower()).any():
                idx = m[m == name.lower()].index[0]
                for other in df.columns:
                    if other != c:
                        v = pd.to_numeric(df.loc[idx, other], errors="coerce")
                        if pd.notna(v):
                            return float(v)
    return np.nan

val_metric = get_metric(metrics_df, primary)
top_feature = "not available (clustering model has no feature importance)"
if hasattr(est, "feature_importances_"):
    imp = np.asarray(est.feature_importances_).ravel()
    if imp.size and imp.size == len(feats):
        top_feature = feats[int(np.argmax(imp))]
elif hasattr(est, "coef_"):
    coef = np.asarray(est.coef_)
    coef = np.abs(coef).sum(axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
    if coef.size == len(feats):
        top_feature = feats[int(np.argmax(coef))]

rows = [
    ("model", "selected_estimator", model_name, model_key),
    ("model", "pipeline_steps", " -> ".join(model.named_steps.keys()) if hasattr(model, "named_steps") else "single estimator", model_key),
    ("model", "hyperparameters", param_text, model_key),
    ("inputs", "train_rows", str(train.shape[0]), "train"),
    ("inputs", "val_rows", str(val.shape[0]), "val"),
    ("inputs", "feature_count", str(len(feats)), model_key),
    ("inputs", "feature_columns", feat_text, model_key),
    ("validation", "primary_metric", primary, "_protocol"),
    ("validation", "value", f"{val_metric:.4g}" if pd.notna(val_metric) else "NaN", "09_validate_score_validation_FBYY3QH9/validation_metrics.csv"),
    ("limitations", "no_outcome", "Clustering only: no label/outcome column; validation is internal.", "_protocol"),
    ("limitations", "silhouette_scope", "Silhouette is an internal separation score, not external accuracy.", "09_validate_score_validation_FBYY3QH9/validation_metrics.csv"),
    ("limitations", "frozen", "Interpretation only; no refitting or post-validation fitting performed.", model_key),
]
interp = pd.DataFrame(rows, columns=["section", "item", "value", "source"])
interp["value"] = interp["value"].astype(str)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"train rows={train.shape[0]}, val rows={val.shape[0]}")
print(f"model={model_name}, features={len(feats)}")
print(f"{primary}={val_metric:.4g}" if pd.notna(val_metric) else f"{primary}=NA")
print(f"top_feature={top_feature}")
