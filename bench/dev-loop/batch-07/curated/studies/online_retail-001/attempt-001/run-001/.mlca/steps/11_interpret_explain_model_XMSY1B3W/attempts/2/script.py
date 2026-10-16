import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol.get("task", {})
primary_metric = task.get("primary_metric", "silhouette")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

selector = load_pkl(inputs["06_features_retain_features_79PZREQ1/selector_all.pkl"])
model = load_pkl(inputs["latest_model"])
k_sel = pd.read_csv(inputs["08_model_train_alternative_3C6W5R97/k_selection.csv"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_T9RQ4KJY/validation_metrics.csv"])

def get_final_estimator(m):
    if isinstance(m, Pipeline):
        return m.named_steps[list(m.named_steps.keys())[-1]]
    return m

final_model = get_final_estimator(model)
model_type = type(final_model).__name__
params = final_model.get_params()

def get_feature_names(obj, n_features):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except:
            pass
    return [f"f{i}" for i in range(n_features)]

def get_selector_support(sel):
    if isinstance(sel, Pipeline):
        last_step = sel.named_steps[list(sel.named_steps.keys())[-1]]
        return get_selector_support(last_step)
    if hasattr(sel, "get_support"):
        return sel.get_support()
    return None

support = get_selector_support(selector)
if support is not None:
    n_features = len(support)
    feature_names = get_feature_names(selector, n_features)
    selected_features = [f for f, s in zip(feature_names, support) if s]
else:
    selected_features = []

top_feature = None
if hasattr(final_model, "cluster_centers_"):
    centers = final_model.cluster_centers_
    variances = np.var(centers, axis=0)
    if len(variances) == len(selected_features):
        top_idx = np.argmax(variances)
        top_feature = selected_features[top_idx]
    else:
        top_feature = selected_features[0] if selected_features else None
else:
    top_feature = selected_features[0] if selected_features else None

def round_sig(x, sig=4):
    if pd.isna(x) or not isinstance(x, (int, float)):
        return x
    return float(f"{x:.{sig}g}")

val_score = np.nan
if primary_metric in val_metrics.columns:
    val_score = val_metrics[primary_metric].iloc[0]
elif 'metric' in val_metrics.columns and 'value' in val_metrics.columns:
    row = val_metrics[val_metrics['metric'] == primary_metric]
    if not row.empty:
        val_score = row['value'].iloc[0]

best_k = np.nan
if 'silhouette' in k_sel.columns:
    best_row = k_sel.loc[k_sel['silhouette'].idxmax()]
    best_k = best_row.get('k', best_row.get('n_clusters', np.nan))

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

rows = [
    {"aspect": "model_type", "detail": "final estimator class", "value": model_type, "source": "latest_model"},
    {"aspect": "hyperparameters", "detail": "n_clusters", "value": params.get("n_clusters", np.nan), "source": "latest_model"},
    {"aspect": "hyperparameters", "detail": "init", "value": params.get("init", np.nan), "source": "latest_model"},
    {"aspect": "features", "detail": "number selected", "value": len(selected_features), "source": "06_features_retain_features_79PZREQ1/selector_all.pkl"},
    {"aspect": "features", "detail": "top feature", "value": top_feature, "source": "06_features_retain_features_79PZREQ1/selector_all.pkl"},
    {"aspect": "validation", "detail": primary_metric, "value": round_sig(val_score), "source": "09_validate_score_validation_T9RQ4KJY/validation_metrics.csv"},
    {"aspect": "k_selection", "detail": "best k", "value": round_sig(best_k), "source": "08_model_train_alternative_3C6W5R97/k_selection.csv"},
    {"aspect": "inputs", "detail": "train rows", "value": len(train_df), "source": "train.parquet"},
    {"aspect": "inputs", "detail": "val rows", "value": len(val_df), "source": "val.parquet"},
    {"aspect": "limitations", "detail": "unsupervised", "value": "No outcome column; clustering evaluated by internal metric only.", "source": "_protocol"},
]

interp_df = pd.DataFrame(rows)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Interpretation of frozen clustering model")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Model: {model_type}")
print(f"Selected features: {len(selected_features)}")
print(f"Top feature: {top_feature}")
print(f"Validation {primary_metric}: {val_score:.4f}" if not np.isnan(val_score) else "Validation metric: N/A")
print("Interpretation saved to interpretation.csv")
