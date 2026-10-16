import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing '{k}' in _protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
primary_metric = task.get("primary_metric", "reconstruction_error")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

with open(inputs["05_preprocess_fit_preprocessing_E2NYE0HC/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_8M4G64VT/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["07_model_train_baseline_JYP4DTKE/model.pkl"], "rb") as f:
    model = pickle.load(f)
with open(inputs["08_model_train_alternative_29T9VJ6F/alternative_model.pkl"], "rb") as f:
    alt_model = pickle.load(f)

val_metrics = pd.read_csv(inputs["09_validate_score_validation_07PQJRF7/validation_metrics.csv"])

def final_est(m):
    if isinstance(m, Pipeline):
        return m.named_steps[list(m.named_steps.keys())[-1]]
    return m

final_model = final_est(model)
final_alt = final_est(alt_model)

rows = []
rows.append({"aspect": "task_type", "detail": str(task.get("type", "")), "source": "_protocol"})
rows.append({"aspect": "primary_metric", "detail": str(primary_metric), "source": "_protocol"})
rows.append({"aspect": "n_features", "detail": str(len(feature_cols)), "source": "train.parquet"})
rows.append({"aspect": "feature_names", "detail": ", ".join(feature_cols[:10]) + ("..." if len(feature_cols) > 10 else ""), "source": "train.parquet"})

rows.append({"aspect": "model_type", "detail": type(final_model).__name__, "source": "07_model_train_baseline_JYP4DTKE/model.pkl"})
if hasattr(final_model, "get_params"):
    params = final_model.get_params()
    key_params = {k: v for k, v in params.items() if k in ["n_components", "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init", "max_iter", "batch_size"]}
    rows.append({"aspect": "hyperparameters", "detail": json.dumps(key_params, default=str), "source": "07_model_train_baseline_JYP4DTKE/model.pkl"})

rows.append({"aspect": "alternative_model_type", "detail": type(final_alt).__name__, "source": "08_model_train_alternative_29T9VJ6F/alternative_model.pkl"})
if hasattr(final_alt, "get_params"):
    alt_params = final_alt.get_params()
    key_alt = {k: v for k, v in alt_params.items() if k in ["n_components", "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init", "max_iter", "batch_size"]}
    rows.append({"aspect": "alternative_hyperparameters", "detail": json.dumps(key_alt, default=str), "source": "08_model_train_alternative_29T9VJ6F/alternative_model.pkl"})

if not val_metrics.empty:
    for _, r in val_metrics.iterrows():
        rows.append({"aspect": f"validation_{r.get('metric', 'metric')}", "detail": str(r.get('value', '')), "source": "09_validate_score_validation_07PQJRF7/validation_metrics.csv"})

rows.append({"aspect": "limitation", "detail": "No outcome column; unsupervised representation learning only.", "source": "_protocol"})
rows.append({"aspect": "limitation", "detail": "Reconstruction error is the only validation metric; no downstream task performance measured.", "source": "09_validate_score_validation_07PQJRF7/validation_metrics.csv"})

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

top_feature = None
if hasattr(final_model, "components_"):
    loadings = np.abs(final_model.components_).sum(axis=0)
    if len(loadings) == len(feature_cols):
        top_feature = feature_cols[int(np.argmax(loadings))]
elif hasattr(final_model, "coefs_"):
    w = np.abs(final_model.coefs_[0]).sum(axis=1)
    if len(w) == len(feature_cols):
        top_feature = feature_cols[int(np.argmax(w))]

if top_feature is not None:
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"features: {len(feature_cols)}")
print(f"model: {type(final_model).__name__}")
print(f"val metrics rows: {len(val_metrics)}")
print(f"top_feature: {top_feature}")
