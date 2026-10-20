import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
try:
    primary = protocol["task"]["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing {e}; top-level keys: {list(protocol)}")
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

def load_pkl(key):
    with open(paths[key], "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("05_preprocess_fit_preprocessing_TEP20BJ6/preprocessor.pkl")
selector = load_pkl("06_features_retain_features_HRWJ3W8S/selector_all.pkl")
baseline = load_pkl("07_model_train_baseline_4QYJGXS4/model.pkl")
alternative = load_pkl("08_model_train_alternative_W0P9YC9A/alternative_model.pkl")
candidate_models = {"baseline": baseline, "alternative": alternative}

def clean_pipeline(model):
    if isinstance(model, Pipeline):
        return model
    steps = []
    sel_steps = selector.named_steps if isinstance(selector, Pipeline) else {}
    pre_in_sel = any("preprocess" in k.lower() or "preproc" in k.lower() for k in sel_steps)
    if preprocessor is not None and not pre_in_sel:
        steps.append(("preprocessor", preprocessor))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("cluster", model))
    return Pipeline(steps)

pipelines = {k: clean_pipeline(v) for k, v in candidate_models.items()}

val = pd.read_csv(paths["09_validate_score_validation_BJKNFAPK/validation_metrics.csv"])
def to_key(name):
    s = str(name).lower()
    if "alt" in s or "latest" in s:
        return "alternative"
    if "base" in s:
        return "baseline"
    return s

if {"model", "metric", "value"}.issubset(val.columns):
    v = val[val["metric"].astype(str) == primary].copy()
    if v.empty:
        raise KeyError(f"No validation rows for {primary}; available: {sorted(val['metric'].astype(str).unique())}")
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    v["cand"] = v["model"].map(to_key)
    v = v[v["cand"].isin(pipelines)]
    if v.empty:
        raise KeyError(f"No candidate rows for {primary}; models: {sorted(val['model'].astype(str).unique())}")
    agg = v.groupby("cand")["value"].mean()
    best = agg.idxmin() if primary == "davies_bouldin" else agg.idxmax()
elif "model" in val.columns and primary in val.columns:
    v = val.assign(cand=val["model"].map(to_key))
    v = v[v["cand"].isin(pipelines)]
    if v.empty:
        raise KeyError(f"No candidate rows for {primary}; models: {sorted(val['model'].astype(str).unique())}")
    agg = pd.to_numeric(v.set_index("cand")[primary], errors="coerce")
    best = agg.idxmin() if primary == "davies_bouldin" else agg.idxmax()
else:
    raise KeyError(f"validation_metrics.csv lacks model/metric/value or {primary}; columns: {list(val.columns)}")

pipe = pipelines[best]
train = pd.read_parquet(paths["train"])
test = pd.read_parquet(paths["test"])

def feature_cols(df):
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    cols = []
    n = len(df)
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if s.nunique(dropna=False) == n and (not pd.api.types.is_numeric_dtype(s) or any(t in str(c).lower() for t in ["id", "key", "code"])):
            continue
        cols.append(c)
    return cols

def raw_cols_for(p):
    if hasattr(p, "feature_names_in_"):
        return list(p.feature_names_in_)
    if isinstance(p, Pipeline):
        first = p.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return feature_cols(train)

raw_cols = raw_cols_for(pipe)
X_train_raw = train[raw_cols].copy()
X_test_raw = test[raw_cols].copy()
train_labels = pipe.predict(X_train_raw)
test_labels = pipe.predict(X_test_raw)

def transform_input(p, X):
    if isinstance(p, Pipeline):
        last = p.steps[-1][1]
        if hasattr(last, "predict") and not hasattr(last, "transform"):
            return X if len(p.steps) == 1 else p[:-1].transform(X)
        if hasattr(p, "transform"):
            return p.transform(X)
    return X

X_test_rep = transform_input(pipe, X_test_raw)
if hasattr(X_test_rep, "toarray"):
    X_test_rep = X_test_rep.toarray()

n_test = len(X_test_rep)
n_clusters = len(np.unique(test_labels))
if n_clusters < 2 or n_clusters >= n_test:
    sil = ch = db = np.nan
else:
    ss = min(10000, n_test)
    sil = silhouette_score(X_test_rep, test_labels, sample_size=ss, random_state=42)
    ch = calinski_harabasz_score(X_test_rep, test_labels)
    db = davies_bouldin_score(X_test_rep, test_labels)

metrics = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
if primary not in metrics:
    raise KeyError(f"Unsupported primary metric {primary}; available: {list(metrics)}")
primary_value = float(metrics[primary])

metrics_df = pd.DataFrame([
    {"model": best, "metric": k, "value": (float(f"{v:.4g}") if isinstance(v, float) and np.isfinite(v) else v)}
    for k, v in metrics.items()
])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test clustering metrics for validation-selected candidate")
mlca_runtime.record_metric(primary, primary_value, "test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")

assign_df = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": train_labels}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": test_labels}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign_df, "assignments.parquet")

print(f"selected={best}")
print(f"train_rows={len(train)} test_rows={len(test)}")
print(f"{primary}={primary_value:.4g}")
