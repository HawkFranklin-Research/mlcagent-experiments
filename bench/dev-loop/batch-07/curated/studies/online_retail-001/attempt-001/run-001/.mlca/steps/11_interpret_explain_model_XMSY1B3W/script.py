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

def final_est(m):
    if isinstance(m, Pipeline):
        return m.named_steps[list(m.named_steps.keys())[-1]]
    return m

final_model = final_est(model)
model_type = type(final_model).__name__
params = final_model.get_params()

def feat_names(obj, n):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    return [f"f{i}" for i in range(n)]

def support_of(sel):
    if isinstance(sel, Pipeline):
        return support_of(sel.named_steps[list(sel.named_steps.keys())[-1]])
    if hasattr(sel, "get_support"):
        return sel.get_support()
    return None

support = support_of(selector)
if support is not None:
    names = feat_names(selector, len(support))
    selected_features = [f for f, s in zip(names, support) if s]
else:
    selected_features = []

top_feature = None
if hasattr(final_model, "cluster_centers_") and selected_features:
    centers = final_model.cluster_centers_
    if centers.shape[1] == len(selected_features):
        top_feature = selected_features[int(np.argmax(np.var(centers, axis=0)))]
    else:
        top_feature = selected_features[0]
elif selected_features:
    top_feature = selected_features[0]

def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4g}"
    return str(x)

val_score = np.nan
if primary_metric in val_metrics.columns:
    val_score = val_metrics[primary_metric].iloc[0]
elif {"metric", "value"}.issubset(val_metrics.columns):
    row = val_metrics[val_metrics["metric"] == primary_metric]
    if not row.empty:
        val_score = row["value"].iloc[0]

best_k = np.nan
if "silhouette" in k_sel.columns:
    br = k_sel.loc[k_sel["silhouette"].idxmax()]
    best_k = br.get("k", br.get("n_clusters", np.nan))

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

rows = [
    ("model_type", "final estimator class", model_type, "latest_model"),
    ("hyperparameters", "n_clusters", params.get("n_clusters", np.nan), "latest_model"),
    ("hyperparameters", "init", params.get("init", np.nan), "latest_model"),
    ("features", "number selected", len(selected_features), "06_features_retain_features_79PZREQ1/selector_all.pkl"),
    ("features", "top feature", top_feature, "06_features_retain_features_79PZREQ1/selector_all.pkl"),
    ("validation", primary_metric, val_score, "09_validate_score_validation_T9RQ4KJY/validation_metrics.csv"),
    ("k_selection", "best k", best_k, "08_model_train_alternative_3C6W5R97/k_selection.csv"),
    ("inputs", "train rows", len(train_df), "train.parquet"),
    ("inputs", "val rows", len(val_df), "val.parquet"),
    ("limitations", "unsupervised", "No outcome column; clustering evaluated by internal metric only.", "_protocol"),
]
interp_df = pd.DataFrame(
    [{"aspect": a, "detail": d, "value": fmt(v), "source": s} for a, d, v, s in rows]
)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Interpretation of frozen clustering model")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Model: {model_type}")
print(f"Selected features: {len(selected_features)}")
print(f"Top feature: {top_feature}")
print(f"Validation {primary_metric}: {fmt(val_score)}")
print("Interpretation saved to interpretation.csv")
