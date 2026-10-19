import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(proto)}")
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task)}")
primary = task["primary_metric"]
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column") if "group_column" in proto else None
drop = set(targets) | set(excluded) | ({group} if group else set())

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
candidate_paths = {
    "baseline": inputs["07_model_train_baseline_9FVS2S2P/model.pkl"],
    "alternative": inputs["08_model_train_alternative_D2KDW4KS/alternative_model.pkl"],
}
models = {name: pickle.loads(path.read_bytes()) for name, path in candidate_paths.items()}

def input_columns(obj):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    if isinstance(obj, Pipeline) and obj.steps:
        cols = getattr(obj.steps[0][1], "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    return None

first_model = next(iter(models.values()))
feature_cols = input_columns(first_model)
if feature_cols is None:
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.is_unique and (not pd.api.types.is_numeric_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing)):
            continue
        feature_cols.append(c)
else:
    bad = set(feature_cols) & drop
    if bad:
        raise ValueError(f"Saved model input includes excluded/target columns: {sorted(bad)}")
for name, model in models.items():
    cols = input_columns(model)
    if cols is not None and list(cols) != feature_cols:
        raise ValueError(f"{name} expects different input columns")
missing_cols = [c for c in feature_cols if c not in val.columns]
if missing_cols:
    raise KeyError(f"Validation is missing feature columns: {missing_cols}")
X_val = val[feature_cols].copy()

def labels_and_transformed(model, X):
    if isinstance(model, Pipeline):
        labels = model.predict(X)
        if len(model.steps) > 1:
            Xt = model[:-1].transform(X)
        else:
            Xt = X.to_numpy()
    else:
        labels = model.predict(X)
        Xt = X.to_numpy()
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(labels).ravel(), np.asarray(Xt)

metrics = {}
for name, model in models.items():
    labels, Xt = labels_and_transformed(model, X_val)
    n = Xt.shape[0]
    if len(np.unique(labels)) < 2:
        metrics[name] = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
        continue
    sample_size = min(10000, n)
    metrics[name] = {
        "silhouette": silhouette_score(Xt, labels, sample_size=sample_size, random_state=42),
        "calinski_harabasz": calinski_harabasz_score(Xt, labels),
        "davies_bouldin": davies_bouldin_score(Xt, labels),
    }
if primary not in next(iter(metrics.values())):
    raise KeyError(f"Unsupported primary metric {primary!r}; computed {sorted(next(iter(metrics.values())))}")

rows = []
for name, vals in metrics.items():
    for metric_name, value in vals.items():
        v = float(value)
        rows.append({"model": name, "metric": metric_name, "value": float(f"{v:.4g}") if np.isfinite(v) else np.nan})
metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation clustering metrics")

finite = {name: float(vals[primary]) for name, vals in metrics.items() if np.isfinite(vals[primary])}
if not finite:
    raise ValueError("No finite primary validation metric")
if primary == "davies_bouldin":
    best_model = min(finite, key=finite.get)
else:
    best_model = max(finite, key=finite.get)
best_exact = finite[best_model]
best_csv = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["metric"] == primary)]["value"].iloc[0]
mlca_runtime.record_metric(primary, best_exact, "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_csv), source="validation_metrics.csv")

print(f"train rows={len(train)}, val rows={len(val)}, features={len(feature_cols)}")
print(f"validation metric rows={len(metrics_df)}")
print(f"best={best_model} {primary}={best_exact:.4g}")
