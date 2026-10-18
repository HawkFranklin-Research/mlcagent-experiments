import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
task = need(protocol, "task")
primary_metric = need(task, "primary_metric")
group_col = protocol.get("group_column", None)

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def key_like(s):
    if not pd.api.types.is_numeric_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

base_cols = [
    c for c in train.columns
    if c not in set(targets + excluded + ([group_col] if group_col else []))
    and not key_like(train[c])
]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def input_cols(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline) and model.steps:
        first = model.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return base_cols

def transform_and_label(model, df):
    cols = input_cols(model)
    missing = [c for c in cols if c not in df.columns]
    bad = set(cols) & set(targets + excluded + ([group_col] if group_col else []))
    if missing or bad:
        raise ValueError(f"Invalid candidate input columns; missing={missing}, forbidden={sorted(bad)}")
    X_raw = df[cols]
    if not isinstance(model, Pipeline):
        raise TypeError("Saved candidate is not a Pipeline; cannot safely transform raw validation rows.")
    pre = model[:-1]
    Xt = X_raw if len(pre.steps) == 0 else pre.transform(X_raw)
    labels = model.predict(X_raw)
    return Xt, labels

def safe_metric(name, X, labels):
    try:
        if name == "silhouette":
            n = len(labels)
            if n < 2 or len(np.unique(labels)) < 2:
                return np.nan
            return silhouette_score(
                X, labels, sample_size=min(10000, n), random_state=42
            )
        if len(np.unique(labels)) < 2:
            return np.nan
        if name == "calinski_harabasz":
            return calinski_harabasz_score(X, labels)
        if name == "davies_bouldin":
            return davies_bouldin_score(X, labels)
    except Exception:
        return np.nan
    return np.nan

def round4(x):
    return float(f"{float(x):.4g}") if np.isfinite(x) else np.nan

candidates = [
    ("baseline", load_pkl(paths["07_model_train_baseline_MAK3HTGT/model.pkl"])),
    ("alternative", load_pkl(paths["08_model_train_alternative_5AQ82GP1/alternative_model.pkl"])),
]
metric_names = ["silhouette", "calinski_harabasz", "davies_bouldin"]
rows = []
for model_name, model in candidates:
    Xt, labels = transform_and_label(model, val)
    for m in metric_names:
        rows.append({
            "model": model_name,
            "metric": m,
            "value": round4(safe_metric(m, Xt, labels)),
        })

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation clustering metrics.")

higher_better = {"silhouette": True, "calinski_harabasz": True, "davies_bouldin": False}
if primary_metric not in higher_better:
    raise ValueError(f"Unsupported primary metric: {primary_metric}")
primary_rows = metrics_df[
    (metrics_df["metric"] == primary_metric) & metrics_df["value"].notna()
]
if primary_rows.empty:
    raise ValueError("No finite validation scores for the declared primary metric.")
idx = primary_rows["value"].idxmax() if higher_better[primary_metric] else primary_rows["value"].idxmin()
best_model = str(primary_rows.loc[idx, "model"])
best_score = float(primary_rows.loc[idx, "value"])

mlca_runtime.record_metric(primary_metric, best_score, split="val")
mlca_runtime.record_fact_candidate(
    "best_validation_score", best_score, source="validation_metrics.csv"
)
mlca_runtime.record_fact_candidate(
    "best_model", best_model, source="validation_metrics.csv"
)

print(f"val rows: {len(val)}")
print(f"candidates: {[n for n, _ in candidates]}")
print(f"best {primary_metric}: {best_model} {best_score:.4g}")
