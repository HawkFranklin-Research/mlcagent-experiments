import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())
try:
    task = protocol["task"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"missing protocol key {e}; available top-level keys={list(protocol.keys())}")

targets = set(protocol.get("targets", []) or [])
excluded = set(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

features = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or
            pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    features.append(c)
if not features:
    raise ValueError(f"no feature columns; protocol keys={list(protocol.keys())}")
Xval = val[features].copy()

def load(key):
    p = inputs.get(key)
    if p is None:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load("05_preprocess_fit_preprocessing_SYRRSV3B/preprocessor.pkl")
sel = load("06_features_retain_features_4JEKVG72/selector_all.pkl")
mods = {
    "baseline": load("07_model_train_baseline_SAV13199/model.pkl"),
    "alternative": load("08_model_train_alternative_SQA69R0G/alternative_model.pkl"),
}
mods = {k: v for k, v in mods.items() if v is not None}
latest = inputs.get("latest_model")
used = {Path(inputs[k]).resolve() for k in
        ("07_model_train_baseline_SAV13199/model.pkl", "08_model_train_alternative_SQA69R0G/alternative_model.pkl")
        if inputs.get(k)}
if latest is not None and Path(latest).resolve() not in used:
    obj = load("latest_model")
    if obj is not None:
        mods["latest"] = obj
if not mods:
    raise ValueError("no saved candidate models found")

def cols(obj):
    c = getattr(obj, "feature_names_in_", None)
    if c is None and isinstance(obj, Pipeline):
        for st in obj.named_steps.values():
            c = getattr(st, "feature_names_in_", None)
            if c is not None:
                break
    return list(c) if c is not None else None

def transform(obj, X):
    c = cols(obj)
    if c is not None and isinstance(X, pd.DataFrame) and all(x in X.columns for x in c):
        X = X[c]
    Z = obj.transform(X)
    if not isinstance(Z, pd.DataFrame) and hasattr(obj, "get_feature_names_out"):
        try:
            Z = pd.DataFrame(Z, columns=obj.get_feature_names_out(), index=X.index)
        except Exception:
            pass
    return Z

def bare_matrix():
    Xm = Xval
    if isinstance(sel, Pipeline):
        sc = cols(sel)
        if sc is not None and all(c in features for c in sc):
            return transform(sel, Xm)
    if pre is not None:
        Xm = transform(pre, Xm)
    if sel is not None:
        Xm = transform(sel, Xm) if isinstance(Xm, pd.DataFrame) else sel.transform(Xm)
    return Xm

def labels_matrix(model):
    if isinstance(model, Pipeline):
        labels = model.predict(Xval)
        Xm = transform(Pipeline(model.steps[:-1]), Xval) if len(model.steps) > 1 else Xval
    else:
        Xm = bare_matrix()
        labels = model.predict(Xm)
    return np.asarray(labels).ravel(), np.asarray(Xm)

metrics = {}
for name, model in mods.items():
    labels, Xm = labels_matrix(model)
    n, k = len(labels), len(np.unique(labels))
    if n < 2 or k < 2 or k >= n:
        continue
    sample = min(10000, n) if n > 10000 else None
    metrics[name] = {
        "silhouette": silhouette_score(Xm, labels, sample_size=sample, random_state=42),
        "calinski_harabasz": calinski_harabasz_score(Xm, labels),
        "davies_bouldin": davies_bouldin_score(Xm, labels),
    }
if not metrics:
    raise ValueError("no valid clustering candidate on validation")

rows = [{"model": n, "metric": m, "value": float(v[m])}
        for n, v in metrics.items() for m in ("silhouette", "calinski_harabasz", "davies_bouldin")]
df = pd.DataFrame(rows)
df["value"] = df["value"].map(lambda x: float(f"{x:.4g}"))
metric_rows = df[df["metric"] == primary]
if metric_rows.empty:
    raise ValueError(f"primary metric {primary!r} not computed")
if primary == "davies_bouldin":
    best = metric_rows.loc[metric_rows["value"].idxmin()]
else:
    best = metric_rows.loc[metric_rows["value"].idxmax()]
best_name, best_score = str(best["model"]), float(best["value"])

mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
print(f"val_rows={len(val)} candidates={len(metrics)} best={best_name}")
print(f"{primary}={best_score:.4g}")
