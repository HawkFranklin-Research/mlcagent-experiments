import mlca_runtime
import json, pickle
from pathlib import Path
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
    group = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}")
primary = task["primary_metric"]

val = pd.read_parquet(inp["val"])

def load_pkl(key):
    return pickle.loads(Path(inp[key]).read_bytes())

pre = load_pkl("05_preprocess_fit_preprocessing_MATQH9FA/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_5RGESHAY/selector_all.pkl")
cands = {
    "baseline": load_pkl("07_model_train_baseline_M82AMBVM/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_DY0NT3JN/alternative_model.pkl"),
}

def names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        return names(obj.steps[0][1])
    return None

def frontend():
    n_sel, n_pre = names(sel), names(pre)
    if isinstance(sel, Pipeline) and n_sel is not None and n_pre is not None and n_sel == n_pre:
        return sel
    return Pipeline([("preprocessor", pre), ("selector", sel)])

def raw_cols(cand):
    cols = names(cand)
    if cols is not None:
        return cols
    if isinstance(cand, Pipeline) and cand.steps:
        cols = names(cand.steps[0][1])
        if cols is not None:
            return cols
    cols = names(frontend())
    if cols is not None:
        return cols
    drop = set(targets) | set(excluded) | ({group} if group else set())
    return [c for c in val.columns if c not in drop]

def dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

results, best = [], None
for name, cand in cands.items():
    cols = raw_cols(cand)
    bad = set(cols) & (set(targets) | set(excluded) | ({group} if group else set()))
    if bad:
        raise ValueError(f"{name} expects excluded/group columns: {sorted(bad)}")
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise ValueError(f"{name} missing val columns: {missing}")
    Xraw = val[cols].copy()
    if isinstance(cand, Pipeline) and len(cand.steps) >= 2:
        Xt = dense(Pipeline(cand.steps[:-1]).transform(Xraw))
        labels = np.asarray(cand.predict(Xraw))
    else:
        fe = frontend()
        Xt = dense(fe.transform(Xraw))
        labels = np.asarray(cand.predict(Xt))
    n = len(labels)
    sil = np.nan
    if len(np.unique(labels)) >= 2:
        try:
            sil = float(silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42))
        except Exception:
            pass
    try:
        ch = float(calinski_harabasz_score(Xt, labels)) if len(np.unique(labels)) >= 2 else np.nan
    except Exception:
        ch = np.nan
    try:
        db = float(davies_bouldin_score(Xt, labels)) if len(np.unique(labels)) >= 2 else np.nan
    except Exception:
        db = np.nan
    metrics = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
    for m, v in metrics.items():
        results.append({"model": name, "metric": m, "value": v})
    pv = metrics[primary]
    if not np.isnan(pv):
        if best is None:
            best = (name, pv)
        else:
            higher = primary != "davies_bouldin"
            if (higher and pv > best[1]) or (not higher and pv < best[1]):
                best = (name, pv)

df = pd.DataFrame(results)
df["value"] = df["value"].apply(lambda x: x if pd.isna(x) else float(f"{x:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")
if best is None:
    raise ValueError("No valid candidate produced the primary metric")
best_name, best_val = best
mlca_runtime.record_metric(primary, float(best_val), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_val), source="validation_metrics.csv")
print(f"val_rows={len(val)} candidates={list(cands)} best={best_name} {primary}={best_val:.4g}")
