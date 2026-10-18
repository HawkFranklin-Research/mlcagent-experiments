import mlca_runtime, json, pickle, re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
if "task" not in protocol or "primary_metric" not in protocol["task"]:
    raise RuntimeError(f"task.primary_metric missing; top-level keys: {list(protocol)}")
primary = protocol["task"]["primary_metric"]
if not primary:
    raise RuntimeError(f"Declared primary_metric is empty; task keys: {list(protocol['task'])}")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
targets = set(protocol.get("targets", []) or [])
excluded = set(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")
drop = targets | excluded
if group:
    drop.add(group)

def key_like(col, s):
    lc = str(col).lower()
    if re.search(r"(^|_)(id|key|index|row)(_|$)", lc):
        return True
    if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=False) == len(s):
        return True
    return False

features = [c for c in train.columns if c not in drop and not key_like(c, train[c])]

def load_obj(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "estimator"):
            if k in obj:
                return obj[k]
        raise TypeError(f"saved model dict lacks model/pipeline/estimator: {list(obj)}")
    return obj

specs = [
    ("baseline", "07_model_train_baseline_7GET20JD/model.pkl"),
    ("alternative", "08_model_train_alternative_VX601NWG/alternative_model.pkl"),
]
seen_paths = {paths[k].resolve() for _, k in specs}
if "latest_model" in paths and paths["latest_model"].resolve() not in seen_paths:
    specs.append(("latest", "latest_model"))
candidates = [(name, load_obj(paths[k])) for name, k in specs]

def expected_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline) and model.named_steps:
        first = next(iter(model.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def model_input(model, df):
    cols = expected_features(model) or features
    bad = set(cols) & drop
    if bad:
        raise ValueError(f"model expects excluded/target/group columns: {sorted(bad)}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing model features: {missing}")
    return df.loc[:, cols].copy()

def metric_space(model, X):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        pre = model[:-1]
        if hasattr(pre, "transform"):
            return pre.transform(X)
    return X

def dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)

def metric_value(name, X, labels):
    key = str(name).lower().replace("-", "_")
    if "silhouette" in key:
        sample_size = 10000 if len(labels) > 10000 else None
        return silhouette_score(X, labels, sample_size=sample_size, random_state=42)
    if "calinski" in key:
        return calinski_harabasz_score(X, labels)
    if "davies" in key:
        return davies_bouldin_score(X, labels)
    raise ValueError(f"unsupported metric: {name}")

def sig4(x):
    return float(f"{float(x):.4g}") if np.isfinite(x) else float("nan")

metric_names, seen = [], set()
for m in [primary, "silhouette", "calinski_harabasz", "davies_bouldin"]:
    nk = str(m).lower().replace("-", "_")
    if nk not in seen:
        seen.add(nk)
        metric_names.append(m)

rows, scores = [], {}
for name, model in candidates:
    X = model_input(model, val)
    labels = np.asarray(model.predict(X))
    Xm = dense(metric_space(model, X))
    k = len(np.unique(labels))
    vals = {}
    for m in metric_names:
        if len(labels) < 3 or k < 2:
            v = float("nan")
        else:
            try:
                v = float(metric_value(m, Xm, labels))
            except Exception:
                v = float("nan")
        vals[m] = v
        rows.append({"model": name, "metric": m, "value": sig4(v)})
    scores[name] = vals[primary]

valid = [(n, v) for n, v in scores.items() if np.isfinite(v)]
if not valid:
    raise RuntimeError("No candidate produced a finite validation score")
lower_better = str(primary).lower().replace("-", "_") in {"davies_bouldin", "davies_bouldin_score", "db"}
best_name, best_val = (
    min(valid, key=lambda x: x[1]) if lower_better else max(valid, key=lambda x: x[1])
)

out = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation clustering metrics")
mlca_runtime.record_metric(primary, float(best_val), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_val), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"val rows: {len(val)}; candidates: {len(candidates)}")
print(f"features: {len(features)}")
print(f"best {primary}: {best_val:.4g} ({best_name})")
