import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k}; available top-level keys: {sorted(d)}")
    return d[k]

task = need(protocol, "task")
primary = need(task, "primary_metric")
targets = need(protocol, "targets") or []
excluded = need(protocol, "excluded_columns") or []
group = need(protocol, "group_column")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

ex = set(targets) | set(excluded) | ({group} if group else set())
feature_columns = []
for c in train.columns:
    if c in ex:
        continue
    s = train[c]
    distinct_all = s.nunique(dropna=False) == len(train)
    key_like = distinct_all and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )
    if not key_like:
        feature_columns.append(c)

def expected_columns(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

cand_specs = [
    ("baseline", paths["07_model_train_baseline_XYVYQTXX/model.pkl"]),
    ("alternative", paths["08_model_train_alternative_RH7MFB6Q/alternative_model.pkl"]),
]
seen = {str(p.resolve()) for _, p in cand_specs}
if "latest_model" in paths:
    lp = paths["latest_model"]
    if str(lp.resolve()) not in seen:
        cand_specs.append(("latest", lp))

models = {}
for name, path in cand_specs:
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

def safe_metric(func, *args, **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return np.nan

rows = []
for name, model in models.items():
    cols = expected_columns(model) or feature_columns
    bad = set(cols) & ex
    if bad:
        raise ValueError(f"model {name} expects excluded columns: {sorted(bad)}")
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"model {name} missing validation columns: {missing}")

    X = val.loc[:, cols]
    labels = np.asarray(model.predict(X)).ravel()

    X_metric = None
    try:
        if hasattr(model, "steps"):
            X_metric = model[:-1].transform(X)
        elif hasattr(model, "transform"):
            X_metric = model.transform(X)
    except Exception:
        X_metric = None

    if X_metric is None:
        if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
            raise ValueError(f"model {name} has no transform for non-numeric features")
        X_metric = X.to_numpy()

    n = len(labels)
    if n > 10000:
        idx = np.random.default_rng(42).choice(n, size=10000, replace=False)
        Xs = X_metric.iloc[idx] if hasattr(X_metric, "iloc") else X_metric[idx]
        ls = labels[idx]
    else:
        Xs, ls = X_metric, labels

    if hasattr(Xs, "toarray"):
        Xs = Xs.toarray()
    if hasattr(Xs, "values"):
        Xs = Xs.values

    n_lab = len(np.unique(ls))
    sil = safe_metric(
        silhouette_score, Xs, ls,
        sample_size=min(10000, len(ls)), random_state=42
    ) if 1 < n_lab < len(ls) else np.nan
    ch = safe_metric(calinski_harabasz_score, Xs, ls) if n_lab > 1 else np.nan
    db = safe_metric(davies_bouldin_score, Xs, ls) if n_lab > 1 else np.nan

    rows.extend([
        (name, "silhouette", sil),
        (name, "calinski_harabasz", ch),
        (name, "davies_bouldin", db),
    ])

metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
metrics["value"] = metrics["value"].map(
    lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan
)
mlca_runtime.save_table(
    metrics, "validation_metrics.csv",
    caption="Validation clustering metrics for saved candidates."
)

primary_rows = metrics[metrics["metric"] == primary].dropna(subset=["value"])
if primary_rows.empty:
    raise ValueError(f"No finite validation values for primary metric {primary}")

higher_is_better = primary not in {"davies_bouldin"}
best_row = (
    primary_rows.loc[primary_rows["value"].idxmax()]
    if higher_is_better
    else primary_rows.loc[primary_rows["value"].idxmin()]
)
best_model = str(best_row["model"])
best_score = float(best_row["value"])

mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate(
    "best_model", best_model, source="validation_metrics.csv"
)
mlca_runtime.record_fact_candidate(
    "best_validation_score", best_score, source="validation_metrics.csv"
)

print(f"val rows={len(val)} candidates={len(models)}")
print(f"metric rows={len(metrics)} sample_size={min(10000, len(val))}")
print(f"best {best_model} {primary}={best_score:.4g}")
