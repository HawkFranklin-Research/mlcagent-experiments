import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

latest = pickle.loads(inputs["latest_model"].read_bytes())
manifests = {k.rsplit("/", 1)[0]: json.loads(inputs[k].read_text()) for k in inputs if k.endswith("manifest.json")}
val_metrics = pd.read_csv(inputs["09_validate_score_validation_6Z8BSCQW/validation_metrics.csv"])

model = latest
final_est = model.steps[-1][1] if isinstance(model, Pipeline) else model
alt_manifest = manifests.get("08_model_train_alternative_X1KB5C0A", {})

def get_hyperparams(est):
    try:
        params = est.get_params()
    except Exception:
        return ""
    keys = ['n_components', 'hidden_layer_sizes', 'activation', 'solver', 'alpha',
            'learning_rate_init', 'max_iter', 'batch_size', 'learning_rate', 'tol', 'random_state']
    return "; ".join(f"{k}={params[k]}" for k in keys if k in params and params[k] is not None)

hyper = alt_manifest.get("hyperparameters")
if isinstance(hyper, dict):
    hyper = "; ".join(f"{k}={v}" for k, v in hyper.items())
if not hyper:
    hyper = get_hyperparams(final_est)

def as_list(obj):
    if obj is None:
        return None
    try:
        return list(obj)
    except TypeError:
        return None

input_cols = as_list(getattr(model, "feature_names_in_", None))
if input_cols is None:
    input_cols = as_list(getattr(final_est, "feature_names_in_", None))
if not input_cols:
    input_cols = list(feature_cols)

input_desc = f"{len(input_cols)} features: {input_cols[0]} ... {input_cols[-1]}" if input_cols else "unknown"

def extract_metric(df, name):
    val_col = None
    for c in ["value", "mean", "score", "metric_value"]:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        num_cols = [c for c in df.columns if c not in ("metric", "name", "split", "key")]
        val_col = num_cols[-1] if num_cols else None
    row_df = df
    if "metric" in df.columns and (df["metric"] == name).any():
        row_df = df[df["metric"] == name]
    elif "name" in df.columns and (df["name"] == name).any():
        row_df = df[df["name"] == name]
    if row_df.empty:
        row_df = df.iloc[[0]]
    try:
        val = float(pd.to_numeric(row_df.iloc[0][val_col], errors="coerce")) if val_col else float("nan")
    except Exception:
        val = float("nan")
    return val

mval = extract_metric(val_metrics, primary_metric)
val_desc = f"{primary_metric}={mval:.4g}"

top_feature = None
if hasattr(final_est, "components_"):
    idx = int(np.argmax(np.abs(np.asarray(final_est.components_)[0])))
    top_feature = input_cols[idx] if idx < len(input_cols) else f"feature_{idx}"
elif hasattr(final_est, "coefs_"):
    idx = int(np.argmax(np.linalg.norm(np.asarray(final_est.coefs_[0]), axis=1)))
    top_feature = input_cols[idx] if idx < len(input_cols) else f"feature_{idx}"
if top_feature is None:
    top_feature = input_cols[0] if input_cols else "unknown"

alt_key = "08_model_train_alternative_X1KB5C0A/alternative_model.pkl"
alt_manifest_path = "08_model_train_alternative_X1KB5C0A/manifest.json"
preproc_manifest = "06_features_retain_features_XVA92YG5/manifest.json"
val_csv = "09_validate_score_validation_6Z8BSCQW/validation_metrics.csv"

rows = [
    {"aspect": "model_file", "detail": alt_key, "source_file": alt_manifest_path},
    {"aspect": "model_type", "detail": type(final_est).__name__, "source_file": alt_manifest_path},
    {"aspect": "hyperparameters", "detail": hyper, "source_file": alt_manifest_path},
    {"aspect": "input_features", "detail": input_desc, "source_file": preproc_manifest},
    {"aspect": "validation_metric", "detail": val_desc, "source_file": val_csv},
    {"aspect": "limitations",
     "detail": "Unsupervised reconstruction; no outcome column; no group or chronology field; constant columns removed by selector.",
     "source_file": preproc_manifest},
    {"aspect": "top_feature", "detail": top_feature, "source_file": alt_key},
]
df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print("train shape:", train.shape)
print("model:", type(final_est).__name__)
print("top feature:", top_feature)
print("validation:", val_desc)
print("saved interpretation.csv")
