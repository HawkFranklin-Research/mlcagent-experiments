import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
top = set(protocol)

def req(d, key, where):
    if key not in d:
        raise KeyError(f"Missing {key!r} in {where}; available keys: {sorted(d)}")
    return d[key]

targets = req(protocol, "targets", "top-level")
excluded = req(protocol, "excluded_columns", "top-level")
group_col = protocol.get("group_column", None)
task = req(protocol, "task", "top-level")
primary = req(task, "primary_metric", "task")
forbidden = set(targets) | set(excluded)
if group_col:
    forbidden.add(group_col)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s))

feature_cols = [c for c in train.columns
                if c not in forbidden and not is_key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def obj_cols(obj, default):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return list(default)

def make_predictor(model, prep, sel):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        if all(c in feature_cols for c in cols):
            return model, cols
    sel_cols = obj_cols(sel, feature_cols)
    if all(c in feature_cols for c in sel_cols):
        pipe = Pipeline([("selector", sel), ("model", model)])
        first = sel
    else:
        pipe = Pipeline([("preprocessor", prep), ("selector", sel), ("model", model)])
        first = prep
    return pipe, obj_cols(first, feature_cols)

def find_stability(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "stability" in k.lower() and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            r = find_stability(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_stability(v)
            if r is not None:
                return r
    return None

def sig4(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(f"{float(v):.4g}")

prep = load_pkl(inp["05_preprocess_fit_preprocessing_RXV573Z5/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_6C317WGV/selector_all.pkl"])
candidates = [
    ("baseline", inp["07_model_train_baseline_FFKH7YPA/model.pkl"],
     inp["07_model_train_baseline_FFKH7YPA/manifest.json"]),
    ("alternative", inp["08_model_train_alternative_TEGYDRX5/alternative_model.pkl"],
     inp["08_model_train_alternative_TEGYDRX5/manifest.json"]),
]

rows = []
primary_scores = {}
for name, model_path, manifest_path in candidates:
    model = load_pkl(model_path)
    predictor, cols = make_predictor(model, prep, sel)
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"{name} expects missing validation columns: {missing}")
    X_in = val[cols].copy()
    labels = predictor.predict(X_in)
    X_trans = predictor[:-1].transform(X_in) if isinstance(predictor, Pipeline) and len(predictor.steps) > 1 else X_in
    n = len(labels)
    ss = min(10000, n)
    sil = silhouette_score(X_trans, labels, sample_size=ss, random_state=42)
    ch = calinski_harabasz_score(X_trans, labels)
    db = davies_bouldin_score(X_trans, labels)
    manifest = json.loads(manifest_path.read_text())
    st = find_stability(manifest)
    primary_scores[name] = sil
    for metric, value in [("silhouette", sil), ("calinski_harabasz", ch),
                          ("davies_bouldin", db), ("stability", st)]:
        rows.append({"model": name, "metric": metric, "value": sig4(value)})

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation clustering metrics for saved candidates.")

lower_better = {"davies_bouldin", "davies-bouldin", "dbi"}
if primary.lower() in lower_better:
    best_name = min(primary_scores, key=primary_scores.get)
else:
    best_name = max(primary_scores, key=primary_scores.get)
best_score = float(primary_scores[best_name])

mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score,
                                   source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name,
                                   source="validation_metrics.csv")

print(f"val rows={len(val)} features={len(feature_cols)}")
print(f"best={best_name} {primary}={best_score:.4g}")
print(f"validation_metrics.csv rows={len(metrics_df)}")
