import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

# Read protocol
protocol = json.loads(inputs["_protocol"].read_text())
required = ["task", "targets", "excluded_columns", "group_column"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys: {missing}. Available: {list(protocol.keys())}")

task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

# Load data and artifacts
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_RHW0T9SM/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["07_model_train_baseline_DWY924K5/model.pkl"], "rb") as f:
    model = pickle.load(f)
with open(inputs["08_model_train_alternative_3C755C37/alternative_model.pkl"], "rb") as f:
    alt_model = pickle.load(f)
cluster_sel = pd.read_csv(inputs["07_model_train_baseline_DWY924K5/cluster_selection.csv"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_JM44JZJV/validation_metrics.csv"])

# Helper to get feature names
def get_feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            names = get_feature_names(step)
            if names:
                return names
    return None

feature_cols = get_feature_names(preprocessor) or get_feature_names(selector)
if feature_cols is None:
    drop = set(targets) | set(excluded) | {group_col}
    feature_cols = [c for c in train.columns if c not in drop]

# Get selected features
selected_features = None
if hasattr(selector, "get_support"):
    support = selector.get_support()
    if hasattr(selector, "feature_names_in_"):
        selected_features = [f for f, s in zip(selector.feature_names_in_, support) if s]
    elif feature_cols and len(feature_cols) == len(support):
        selected_features = [f for f, s in zip(feature_cols, support) if s]
elif hasattr(selector, "named_steps"):
    for step in selector.named_steps.values():
        if hasattr(step, "get_support"):
            support = step.get_support()
            if hasattr(step, "feature_names_in_"):
                selected_features = [f for f, s in zip(step.feature_names_in_, support) if s]
            elif feature_cols and len(feature_cols) == len(support):
                selected_features = [f for f, s in zip(feature_cols, support) if s]
            break

# Model info
def get_model_info(m):
    if hasattr(m, "named_steps"):
        final = list(m.named_steps.values())[-1]
        return type(final).__name__, final.get_params()
    return type(m).__name__, m.get_params()

model_type, model_params = get_model_info(model)
alt_type, alt_params = get_model_info(alt_model)

# Cluster selection
best_row = cluster_sel.loc[cluster_sel['silhouette'].idxmax()] if 'silhouette' in cluster_sel.columns else cluster_sel.iloc[0]
best_k = best_row.get('k', best_row.get('n_clusters', None))
best_sil = best_row.get('silhouette', None)

# Validation metrics
val_sil = val_metrics['silhouette'].iloc[0] if 'silhouette' in val_metrics.columns and len(val_metrics) > 0 else None

# Top feature
top_feature = None
centers = None
if hasattr(model, "named_steps"):
    final = list(model.named_steps.values())[-1]
    if hasattr(final, "cluster_centers_"):
        centers = final.cluster_centers_
elif hasattr(model, "cluster_centers_"):
    centers = model.cluster_centers_

if centers is not None:
    var = np.var(centers, axis=0)
    idx = int(np.argmax(var))
    if selected_features and len(selected_features) == centers.shape[1]:
        top_feature = selected_features[idx]
    elif feature_cols and len(feature_cols) == centers.shape[1]:
        top_feature = feature_cols[idx]
    else:
        top_feature = f"feature_{idx}"

if top_feature is None:
    top_feature = (selected_features or feature_cols or ["unknown"])[0]

# Build interpretation table
rows = [
    {"aspect": "model_type", "description": model_type, "source": "07_model_train_baseline_DWY924K5/model.pkl"},
    {"aspect": "alternative_model_type", "description": alt_type, "source": "08_model_train_alternative_3C755C37/alternative_model.pkl"},
    {"aspect": "selected_k", "description": str(best_k), "source": "07_model_train_baseline_DWY924K5/cluster_selection.csv"},
    {"aspect": "best_silhouette", "description": f"{best_sil:.4g}" if best_sil is not None else "NA", "source": "07_model_train_baseline_DWY924K5/cluster_selection.csv"},
    {"aspect": "validation_silhouette", "description": f"{val_sil:.4g}" if val_sil is not None else "NA", "source": "09_validate_score_validation_JM44JZJV/validation_metrics.csv"},
    {"aspect": "n_features", "description": str(len(feature_cols)), "source": "05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl"},
    {"aspect": "n_selected_features", "description": str(len(selected_features)) if selected_features else "NA", "source": "06_features_retain_features_RHW0T9SM/selector_all.pkl"},
    {"aspect": "top_feature", "description": top_feature, "source": "07_model_train_baseline_DWY924K5/model.pkl"},
    {"aspect": "limitations", "description": "No outcome; internal silhouette only; CustomerID groups kept intact; identifiers excluded.", "source": "_protocol"},
]
for k in ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'random_state', 'algorithm']:
    if k in model_params:
        rows.append({"aspect": f"model_param_{k}", "description": str(model_params[k]), "source": "07_model_train_baseline_DWY924K5/model.pkl"})

df_interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(df_interp, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Model: {model_type}, k={best_k}, val_sil={val_sil}")
print(f"Top feature: {top_feature}")
print(f"Saved interpretation.csv with {len(df_interp)} rows")
