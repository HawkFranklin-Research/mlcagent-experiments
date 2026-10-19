import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    primary_metric = protocol["task"]["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing task.primary_metric. Available top-level keys: {list(protocol.keys())}") from e

targets = protocol.get("targets", []) or []
excluded = protocol.get("excluded_columns", []) or []
group = protocol.get("group_column", None)
train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

def is_key_like(s):
    n = s.notna().sum()
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == n
    return False

exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)
feature_cols = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]

cand_paths = {
    "baseline": inputs["07_model_train_baseline_726ZEXFH/model.pkl"],
    "alternative": inputs["08_model_train_alternative_R61594N7/alternative_model.pkl"],
}
try:
    vmetrics = pd.read_parquet(inputs["09_validate_score_validation_3PNV70HF/validation_metrics.parquet"])
except Exception:
    vmetrics = pd.read_csv(inputs["09_validate_score_validation_3PNV70HF/validation_metrics.csv"])
cols = {str(c).lower(): c for c in vmetrics.columns}
model_col = cols.get("model") or cols.get("candidate") or cols.get("name")
metric_col = cols.get("metric") or cols.get("metric_name")
value_col = cols.get("value") or cols.get("score")
if metric_col is None or value_col is None:
    raise ValueError(f"Validation metrics columns not recognized: {list(vmetrics.columns)}")
vprim = vmetrics[vmetrics[metric_col].astype(str).str.lower() == primary_metric.lower()]
if vprim.empty:
    raise ValueError(f"No validation rows for primary metric {primary_metric!r}")
if model_col is not None:
    best_name = str(vprim.groupby(model_col)[value_col].max().idxmax())
else:
    best_name = str(vprim.loc[vprim[value_col].astype(float).idxmax()].name)
sel_key = next((k for k in cand_paths if k in best_name.lower()), None)
if sel_key is None:
    sel_key = "baseline" if "base" in best_name.lower() else "alternative" if "alt" in best_name.lower() else list(cand_paths)[0]

with open(cand_paths[sel_key], "rb") as f:
    model = pickle.load(f)
with open(inputs["05_preprocess_fit_preprocessing_KQAA3HGR/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_SPWDSE6R/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

raw_cols = getattr(model, "feature_names_in_", None)
if raw_cols is None:
    raw_cols = getattr(preprocessor, "feature_names_in_", None)
if raw_cols is None:
    raw_cols = getattr(selector, "feature_names_in_", None)
if raw_cols is None:
    raw_cols = feature_cols
raw_cols = list(raw_cols)

def transform_predict(obj, df, pre, sel, raw_cols):
    X = df[raw_cols].copy()
    if isinstance(obj, Pipeline):
        labels = obj.predict(X)
        try:
            Xt = obj[:-1].transform(X)
        except Exception:
            Xt = X
        return labels, Xt
    Xt = X
    if sel is not None:
        sc = getattr(sel, "feature_names_in_", None)
        if sc is not None and list(sc) == list(Xt.columns):
            Xt = sel.transform(Xt)
        elif pre is not None:
            pc = getattr(pre, "feature_names_in_", None)
            if pc is not None and list(pc) == list(Xt.columns):
                Xt = pre.transform(Xt)
            sc = getattr(sel, "feature_names_in_", None)
            if sc is not None and list(sc) == list(Xt.columns):
                Xt = sel.transform(Xt)
    elif pre is not None:
        pc = getattr(pre, "feature_names_in_", None)
        if pc is not None and list(pc) == list(Xt.columns):
            Xt = pre.transform(Xt)
    return obj.predict(Xt), Xt

labels_train, _ = transform_predict(model, train, preprocessor, selector, raw_cols)
labels_test, X_test = transform_predict(model, test, preprocessor, selector, raw_cols)

def take_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

n = len(test)
if n > 10000:
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, size=10000, replace=False))
else:
    idx = np.arange(n)
Xm = take_rows(X_test, idx)
ym = np.asarray(labels_test)[idx]
try:
    if len(np.unique(ym)) < 2:
        raise ValueError
    sil = silhouette_score(Xm, ym, sample_size=min(10000, len(ym)), random_state=42)
    ch = calinski_harabasz_score(Xm, ym)
    db = davies_bouldin_score(Xm, ym)
except Exception:
    sil = ch = db = np.nan

def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan
metrics = {"silhouette": sig4(sil), "calinski_harabasz": sig4(ch), "davies_bouldin": sig4(db)}
if primary_metric not in metrics:
    metrics[primary_metric] = metrics["silhouette"]
rows = [{"model": sel_key, "metric": primary_metric, "value": metrics[primary_metric]},
        {"model": sel_key, "metric": "calinski_harabasz", "value": metrics["calinski_harabasz"]},
        {"model": sel_key, "metric": "davies_bouldin", "value": metrics["davies_bouldin"]}]
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test clustering metrics for validation-selected candidate")
mlca_runtime.record_metric(primary_metric, float(metrics[primary_metric]), split="test")
mlca_runtime.record_metric("calinski_harabasz", float(metrics["calinski_harabasz"]), split="test")
mlca_runtime.record_metric("davies_bouldin", float(metrics["davies_bouldin"]), split="test")
mlca_runtime.record_fact_candidate("test_score", float(metrics[primary_metric]), source="test_metrics.csv")

assign = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": labels_train}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": labels_test}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")
print(f"selected={sel_key}, {primary_metric}={metrics[primary_metric]:.4g}")
print(f"test_rows={len(test)}, metric_sample={len(idx)}")
