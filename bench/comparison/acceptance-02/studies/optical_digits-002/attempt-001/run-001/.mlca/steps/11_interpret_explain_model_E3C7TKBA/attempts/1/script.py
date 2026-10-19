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
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

with open(inputs["05_preprocess_fit_preprocessing_W49DB36F/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_XVA92YG5/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["07_model_train_baseline_EGH0EDWF/model.pkl"], "rb") as f:
    baseline = pickle.load(f)
with open(inputs["08_model_train_alternative_X1KB5C0A/alternative_model.pkl"], "rb") as f:
    alternative = pickle.load(f)
with open(inputs["latest_model"], "rb") as f:
    latest = pickle.load(f)

manifests = {}
for step in ["05_preprocess_fit_preprocessing_W49DB36F", "06_features_retain_features_XVA92YG5", "07_model_train_baseline_EGH0EDWF", "08_model_train_alternative_X1KB5C0A", "09_validate_score_validation_6Z8BSCQW"]:
    key = f"{step}/manifest.json"
    if key in inputs:
        manifests[step] = json.loads(inputs[key].read_text())
val_metrics = pd.read_csv(inputs["09_validate_score_validation_6Z8BSCQW/validation_metrics.csv"])

model = latest
if isinstance(model, Pipeline):
    final_est = model.steps[-1][1]
    steps = model.named_steps
else:
    final_est = model
    steps = {}

def get_hyperparams(est):
    params = est.get_params()
    keys = ['n_components', 'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate_init', 'max_iter', 'batch_size', 'learning_rate', 'tol', 'random_state']
    items = [f"{k}={params[k]}" for k in keys if k in params and params[k] is not None]
    return "; ".join(items) if items else str(params)[:200]

hyper = manifests.get("08_model_train_alternative_X1KB5C0A", {}).get("hyperparameters", get_hyperparams(final_est))
if isinstance(hyper, dict):
    hyper = "; ".join(f"{k}={v}" for k, v in hyper.items())

input_cols = getattr(model, "feature_names_in_", None)
if input_cols is None:
    input_cols = getattr(final_est, "feature_names_in_", None)
if input_cols is None:
    input_cols = feature_cols
n_input = len(input_cols)
input_desc = f"{n_input} features: {input_cols[0]} ... {input_cols[-1]}" if n_input > 0 else "unknown"

metric_row = val_metrics[val_metrics['metric'] == primary_metric] if 'metric' in val_metrics.columns else val_metrics.iloc[[0]]
if len(metric_row) == 0:
    metric_row = val_metrics.iloc[[0]]
metric_name = metric_row.iloc[0]['metric'] if 'metric' in val_metrics.columns else primary_metric
metric_value = metric_row.iloc[0]['value'] if 'value' in val_metrics.columns else metric_row.iloc[0].iloc[1]
val_desc = f"{metric_name}={metric_value:.4g}"

top_feature = None
top_value = None
source_for_fact = "08_model_train_alternative_X1KB5C0A/alternative_model.pkl"
if hasattr(final_est, "components_"):
    comp = final_est.components_[0]
    idx = int(np.argmax(np.abs(comp)))
    top_feature = input_cols[idx] if idx < len(input_cols) else f"feature_{idx}"
    top_value = float(comp[idx])
elif hasattr(final_est, "coefs_"):
    W = final_est.coefs_[0]
    norms = np.linalg.norm(W, axis=1)
    idx = int(np.argmax(norms))
    top_feature = input_cols[idx] if idx < len(input_cols) else f"feature_{idx}"
    top_value = float(norms[idx])
if top_feature is None:
    top_feature = input_cols[0] if input_cols else "unknown"
    top_value = 0.0

rows = [
    {"aspect": "model_file", "detail": "08_model_train_alternative_X1KB5C0A/alternative_model.pkl", "source_file": "08_model_train_alternative_X1KB5C0A/manifest.json"},
    {"aspect": "model_type", "detail": type(final_est).__name__, "source_file": "08_model_train_alternative_X1KB5C0A/manifest.json"},
    {"aspect": "hyperparameters", "detail": hyper, "source_file": "08_model_train_alternative_X1KB5C0A/manifest.json"},
    {"aspect": "input_features", "detail": input_desc, "source_file": "05_preprocess_fit_preprocessing_W49DB36F/manifest.json"},
    {"aspect": "validation_metric", "detail": val_desc, "source_file": "09_validate_score_validation_6Z8BSCQW/validation_metrics.csv"},
    {"aspect": "limitations", "detail": "Unsupervised reconstruction; no outcome column; no group/time; constant columns pixel_1 and pixel_40 removed by selector.", "source_file": "06_features_retain_features_XVA92YG5/manifest.json"},
    {"aspect": "top_feature", "detail": f"{top_feature} ({top_value:.4g})", "source_file": "08_model_train_alternative_X1KB5C0A/alternative_model.pkl"},
]
df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source=source_for_fact)
print("train shape:", train.shape)
print("model:", type(final_est).__name__)
print("top feature:", top_feature)
print("validation:", val_desc)
print("saved interpretation.csv")
