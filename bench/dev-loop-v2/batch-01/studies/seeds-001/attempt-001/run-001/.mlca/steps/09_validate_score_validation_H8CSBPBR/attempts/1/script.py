import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
    primary_metric = task["primary_metric"]
except KeyError as e:
    raise ValueError(f"Missing protocol key {e}; available top-level keys: {list(protocol)}")
group_col = protocol.get("group_column") if "group_column" in protocol else None
val = pd.read_parquet(inp["val"])

def load_pkl(key):
    with inp[key].open("rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_1E8V578C/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_Y7PF8JJ2/selector_all.pkl")

cand, seen = {}, set()
def add_file(name, key):
    if key not in inp:
        return
    p = inp[key]
    try:
        obj = load_pkl(key)
    except Exception:
        return
    rp = str(p.resolve())
    if rp in seen:
        return
    seen.add(rp)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "predict"):
                cand[f"{name}:{k}"] = v
    elif hasattr(obj, "predict"):
        cand[name] = obj

add_file("baseline", "07_model_train_baseline_SWTMFPFT/model.pkl")
add_file("alternative", "08_model_train_alternative_XQ00QFTN/alternative_model.pkl")
add_file("latest", "latest_model")
if not cand:
    raise ValueError("No saved candidate models with predict found")

def is_key_like(s):
    name = s.name.lower()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return len(s) > 1 and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return any(k in name for k in ("id", "key", "index")) and s.nunique(dropna=True) == len(s)
    return False

def raw_cols_for(model):
    for obj in (model, sel, pre):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            cols = list(obj.feature_names_in_)
            if cols and all(c in val.columns for c in cols):
                return cols
    base = [c for c in val.columns if c not in set(targets) | set(excluded) and c != group_col]
    return [c for c in base if not is_key_like(val[c])]

def preprocess_features(X, pre, sel):
    if sel is not None and hasattr(sel, "feature_names_in_"):
        sc = list(sel.feature_names_in_)
        if all(c in X.columns for c in sc):
            try:
                return sel.transform(X[sc])
            except Exception:
                pass
    Z = X
    if pre is not None:
        pc = list(getattr(pre, "feature_names_in_", X.columns))
        if all(c in X.columns for c in pc):
            try:
                Z = pre.transform(X[pc])
            except Exception:
                Z = X
    if sel is not None:
        try:
            Z = sel.transform(Z)
        except Exception:
            pass
    return Z

def predict_labels(model, X_raw, pre, sel):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        if all(c in X_raw.columns for c in cols):
            return model.predict(X_raw[cols])
    Z = preprocess_features(X_raw, pre, sel)
    n = getattr(model, "n_features_in_", None)
    if n == X_raw.shape[1] and (not hasattr(Z, "shape") or Z.shape[1] != n):
        return model.predict(X_raw)
    return model.predict(Z)

def embed_for(model, X_raw, pre, sel):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        try:
            cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else X_raw.columns
            return model[:-1].transform(X_raw[cols])
        except Exception:
            pass
    Z = preprocess_features(X_raw, pre, sel)
    arr = Z.values if hasattr(Z, "values") else np.asarray(Z)
    if getattr(model, "n_features_in_", None) == X_raw.shape[1] and arr.shape[1] != X_raw.shape[1]:
        return X_raw.values
    return arr

def metric_scores(emb, labels):
    emb = np.asarray(emb)
    labels = np.asarray(labels)
    n, k = len(labels), len(np.unique(labels))
    out = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    if 1 < k < n:
        m = min(10000, n)
        try:
            out["silhouette"] = float(silhouette_score(emb, labels, sample_size=m, random_state=42))
        except Exception:
            pass
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(emb, labels))
        except Exception:
            pass
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(emb, labels))
        except Exception:
            pass
    return out

def primary_value(m, p):
    p = p.lower()
    if "silhouette" in p:
        return m["silhouette"]
    if "calinski" in p or "harabasz" in p:
        return m["calinski_harabasz"]
    if "davies" in p or "bouldin" in p:
        return m["davies_bouldin"]
    return m.get(p, m["silhouette"])

def r4(x):
    return np.nan if pd.isna(x) else float(f"{x:.4g}")

scores, rows = {}, []
for name, model in cand.items():
    cols = raw_cols_for(model)
    if not cols:
        continue
    X_raw = val[cols].copy()
    try:
        labels = predict_labels(model, X_raw, pre, sel)
        emb = embed_for(model, X_raw, pre, sel)
    except Exception:
        continue
    m = metric_scores(emb, labels)
    scores[name] = m
    for metric, value in m.items():
        rows.append({"model": name, "metric": metric, "value": r4(value)})
    if primary_metric not in m:
        rows.append({"model": name, "metric": primary_metric, "value": r4(primary_value(m, primary_metric))})
if not scores:
    raise ValueError("No candidate could be scored on validation")

higher_better = not ("davies" in primary_metric.lower() or "bouldin" in primary_metric.lower())
best_name, best_score = None, None
for name, m in scores.items():
    v = primary_value(m, primary_metric)
    if pd.isna(v):
        continue
    if best_score is None or (v > best_score if higher_better else v < best_score):
        best_name, best_score = name, v
if best_name is None:
    raise ValueError("No valid primary validation score")

table = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(table, "validation_metrics.csv")
mlca_runtime.record_metric(primary_metric, float(best_score), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"val rows={len(val)}, candidates={len(cand)}, scored={len(scores)}")
print(f"best={best_name}")
print(f"{primary_metric}={best_score:.4g}")
