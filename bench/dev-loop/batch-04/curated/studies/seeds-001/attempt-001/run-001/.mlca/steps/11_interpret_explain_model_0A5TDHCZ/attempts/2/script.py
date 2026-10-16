import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
cluster_sel = pd.read_csv(inputs["07_model_train_baseline_BPBM7F9H/cluster_selection.csv"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_B89RX1GQ/validation_metrics.csv"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

model = load_pkl("07_model_train_baseline_BPBM7F9H/model.pkl")
preprocessor = load_pkl("05_preprocess_fit_preprocessing_FVVN700K/preprocessor.pkl")
selector = load_pkl("06_features_retain_features_FW7XY320/selector_all.pkl")

def final_est(m):
    if hasattr(m, "named_steps"):
        return list(m.named_steps.values())[-1]
    return m

est = final_est(model)
params = est.get_params() if hasattr(est, "get_params") else {}
keys = ["n_clusters", "init", "n_init", "max_iter", "random_state", "algorithm"]
hyper = ", ".join(f"{k}={params[k]}" for k in keys if k in params)
n_clusters = params.get("n_clusters", np.nan)
if np.isnan(n_clusters) and "n_clusters" in cluster_sel.columns:
    if "silhouette" in cluster_sel.columns:
        n_clusters = cluster_sel.loc[cluster_sel["silhouette"].idxmax(), "n_clusters"]
    else:
        n_clusters = cluster_sel["n_clusters"].iloc[0]

def get_metric(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return float(df[col].iloc[0])
    if "metric" in df.columns and "value" in df.columns:
        row = df[df["metric"].str.lower() == name.lower()]
        if len(row):
            return float(row["value"].iloc[0])
    return np.nan

sil = get_metric(val_metrics, "silhouette")
db = get_metric(val_metrics, "davies_bouldin")
ch = get_metric(val_metrics, "calinski_harabasz")

def get_feature_names(m, fallback):
    if hasattr(m, "get_feature_names_out"):
        try:
            return list(m.get_feature_names_out())
        except Exception:
            pass
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except Exception:
                    pass
    return list(fallback)

names = get_feature_names(model, train.columns)
if len(names) != len(train.columns):
    names = get_feature_names(selector, train.columns)
if len(names) != len(train.columns):
    names = get_feature_names(preprocessor, train.columns)
if len(names) != len(train.columns):
    names = list(train.columns)

top_feature = names[0] if names else "unknown"
top_score = np.nan
if hasattr(est, "cluster_centers_"):
    centers = np.asarray(est.cluster_centers_)
    if centers.ndim == 2:
        spread = centers.std(axis=0)
        idx = int(np.argmax(spread))
        top_feature = names[idx] if idx < len(names) else f"feature_{idx}"
        top_score = float(spread[idx])
elif hasattr(est, "feature_importances_"):
    imp = np.asarray(est.feature_importances_)
    idx = int(np.argmax(imp))
    top_feature = names[idx] if idx < len(names) else f"feature_{idx}"
    top_score = float(imp[idx])

def r4(x):
    if isinstance(x, (float, np.floating)) and not np.isnan(x):
        return float(f"{x:.4g}")
    return x

rows = [
    ("model", "type", type(est).__name__, np.nan, "model.pkl"),
    ("model", "n_clusters", np.nan, r4(n_clusters), "model.pkl"),
    ("model", "hyperparameters", hyper, np.nan, "model.pkl"),
    ("inputs", "train_rows", np.nan, len(train), "train.parquet"),
    ("inputs", "val_rows", np.nan, len(val), "val.parquet"),
    ("inputs", "n_features", np.nan, len(train.columns), "train.parquet"),
    ("validation", "silhouette", np.nan, r4(sil), "validation_metrics.csv"),
    ("validation", "davies_bouldin", np.nan, r4(db), "validation_metrics.csv"),
    ("validation", "calinski_harabasz", np.nan, r4(ch), "validation_metrics.csv"),
    ("limitations", "no_outcome", "Clustering only; no outcome column", np.nan, "_protocol"),
    ("limitations", "internal_metrics", "Validation evidence is internal only", np.nan, "validation_metrics.csv"),
    ("top_feature", "name", top_feature, np.nan, "model.pkl"),
    ("top_feature", "score", np.nan, r4(top_score), "model.pkl"),
]
interp = pd.DataFrame(rows, columns=["section", "item", "value_text", "value_num", "source"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"model={type(est).__name__} n_clusters={n_clusters}")
print(f"silhouette={sil:.4g}")
print(f"top_feature={top_feature}")
print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"interpretation.csv rows={len(interp)}")
