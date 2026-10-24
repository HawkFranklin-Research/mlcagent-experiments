import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}. Available top-level keys: {list(protocol)}")
    return protocol[key]

targets = need("targets")
excluded = need("excluded_columns")
task = need("task")
if not isinstance(task, dict) or "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric. Available task keys: {list(task) if isinstance(task, dict) else task}")
primary = task["primary_metric"]
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s: pd.Series, name: str) -> bool:
    if s.dtype == object and s.nunique(dropna=True) == len(s):
        return True
    low = str(name).lower()
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        if any(t in low for t in ("id", "key", "row", "index", "code")):
            return True
    return False

exclude = set(targets) | set(excluded)
if group is not None:
    exclude.add(group)
feature_cols = [c for c in train.columns if c not in exclude and not key_like(train[c], c)]

candidates = {}
for name, key in [
    ("baseline", "07_model_train_baseline_PYZ8RSVK/model.pkl"),
    ("alternative", "08_model_train_alternative_3CGWKKCJ/alternative_model.pkl"),
]:
    with open(inputs[key], "rb") as f:
        candidates[name] = pickle.load(f)

rows = []
scores = {}
for name, model in candidates.items():
    cols = list(getattr(model, "feature_names_in_", feature_cols))
    bad = set(cols) & exclude
    if bad:
        raise ValueError(f"{name} expects excluded columns: {sorted(bad)}")
    X_val = val[cols]
    labels = model.predict(X_val)
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        Xt = model[:-1].transform(X_val)
    else:
        Xt = X_val
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    n = len(Xt)
    ss = 10000 if n > 10000 else None
    sil = silhouette_score(Xt, labels, sample_size=ss, random_state=42)
    ch = calinski_harabasz_score(Xt, labels)
    db = davies_bouldin_score(Xt, labels)
    m = {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)}
    scores[name] = m
    for metric, value in m.items():
        rows.append({"model": name, "metric": metric, "value": float(f"{value:.4g}")})

if primary not in {"silhouette", "calinski_harabasz", "davies_bouldin"}:
    raise ValueError(f"Unsupported primary metric: {primary}")
higher = primary in {"silhouette", "calinski_harabasz"}
if higher:
    best_name, best_metrics = max(scores.items(), key=lambda kv: kv[1][primary])
else:
    best_name, best_metrics = min(scores.items(), key=lambda kv: kv[1][primary])
best_score = float(best_metrics[primary])

df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")
mlca_runtime.record_metric(primary, best_score, split="val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"val rows: {len(val)}")
print(f"candidates: {len(candidates)}")
print(f"best {best_name}: {primary}={best_score:.4g}")
