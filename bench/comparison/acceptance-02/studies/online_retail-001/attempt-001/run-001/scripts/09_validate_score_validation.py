import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
primary_name = protocol["task"]["primary_metric"]
targets = protocol.get("targets", protocol.get("task", {}).get("targets", []))
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    n = s.notna().sum()
    if n <= 1 or s.nunique(dropna=True) != n:
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain")

def load_pkl(key):
    p = inputs.get(key)
    return pickle.loads(p.read_bytes()) if p is not None and p.exists() else None

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ("model", "estimator", "pipeline", "best_model"):
            if k in obj:
                return obj[k]
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
    return obj

def align(obj, Z):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None and hasattr(Z, "columns"):
        cols = list(cols)
        if all(c in Z.columns for c in cols):
            return Z[cols]
    return Z

def frame_for_pipeline(model, df):
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        cols = list(cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing pipeline inputs: {missing[:5]}")
        return df[cols]
    return df[feature_cols]

def frontend_transform(X, pre, sel):
    Z = X
    if sel is not None and hasattr(sel, "feature_names_in_"):
        sc = list(sel.feature_names_in_)
        if sc and set(sc).issubset(set(feature_cols)):
            return sel.transform(X[sc])
    if pre is not None:
        Z = align(pre, X)
        Z = pre.transform(Z)
    if sel is not None:
        Z = align(sel, Z) if hasattr(Z, "columns") else Z
        Z = sel.transform(Z)
    return Z

def labels_and_features(model, X, pre, sel):
    if hasattr(model, "named_steps") and len(model.named_steps) > 0:
        labels = model.predict(X)
        Z = X
        for step in list(model.named_steps.values())[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Z = align(step, Z) if hasattr(Z, "columns") else Z
                Z = step.transform(Z)
        return Z, labels
    Z = frontend_transform(X, pre, sel)
    Z = align(model, Z) if hasattr(Z, "columns") else Z
    return Z, model.predict(Z)

def safe_metrics(Z, labels):
    labels = np.asarray(labels).ravel()
    n = len(labels)
    uniq = np.unique(labels[~pd.isna(labels)]) if n else []
    if n < 2 or len(uniq) < 2:
        return np.nan, np.nan, np.nan
    ss = min(10000, n)
    try:
        sil = silhouette_score(Z, labels, sample_size=ss, random_state=42)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(Z, labels)
    except Exception:
        ch = np.nan
    try:
        db = davies_bouldin_score(Z, labels)
    except Exception:
        db = np.nan
    return sil, ch, db

pre = load_pkl("05_preprocess_fit_preprocessing_HR6HVNXD/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_T3RMYHN2/selector_all.pkl")
cand_specs = [
    ("baseline", "07_model_train_baseline_RA5A8N63/model.pkl"),
    ("alternative", "08_model_train_alternative_Q31ASHB4/alternative_model.pkl"),
]
seen, candidates = set(), []
for name, key in cand_specs:
    p = inputs.get(key)
    if p is None or not p.exists():
        continue
    rp = str(p.resolve())
    if rp in seen:
        continue
    seen.add(rp)
    candidates.append((name, p))
if not candidates:
    raise ValueError("No saved candidate models found")

results = []
for name, path in candidates:
    model = unwrap(pickle.loads(path.read_bytes()))
    X = frame_for_pipeline(model, val) if hasattr(model, "named_steps") else val[feature_cols]
    Z, labels = labels_and_features(model, X, pre, sel)
    sil, ch, db = safe_metrics(Z, labels)
    out_sil_name = primary_name if primary_name.lower().startswith("silhouette") else "silhouette"
    results.extend([
        {"model": name, "metric": out_sil_name, "value": sil},
        {"model": name, "metric": "calinski_harabasz", "value": ch},
        {"model": name, "metric": "davies_bouldin", "value": db},
    ])

lower_better = primary_name.lower() in {"davies_bouldin", "davies_bouldin_score", "db"}
primary = [r for r in results if r["metric"] == primary_name and np.isfinite(r["value"])]
if not primary:
    primary = [r for r in results if r["metric"] == "silhouette" and np.isfinite(r["value"])]
if primary:
    best = min(primary, key=lambda r: r["value"]) if lower_better else max(primary, key=lambda r: r["value"])
    best_name, best_score = best["model"], best["value"]
else:
    best_name, best_score = candidates[0][0], np.nan

def sig4(v):
    return v if pd.isna(v) else float(f"{float(v):.4g}")

df = pd.DataFrame(results)
df["value"] = df["value"].map(sig4)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")
best_tbl = sig4(best_score)
mlca_runtime.record_metric(primary_name, float(best_tbl), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_tbl, source="validation_metrics.csv")
print("val rows", len(val), "features", len(feature_cols))
print("candidates", len(candidates))
print("best", best_name, primary_name, best_tbl)
