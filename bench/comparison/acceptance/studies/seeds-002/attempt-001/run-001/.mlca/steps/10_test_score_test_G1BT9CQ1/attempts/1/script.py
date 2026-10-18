import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise ValueError(f"Missing _protocol key {k}; available top-level keys: {list(protocol)}")
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
features = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    features.append(c)
X_train = train[features].copy()

vm = pd.read_csv(inputs["09_validate_score_validation_C8CS9EB3/validation_metrics.csv"])
try:
    cs = pd.read_parquet(inputs["07_model_train_baseline_MAK3HTGT/cluster_selection.parquet"])
except Exception:
    cs = pd.read_csv(inputs["07_model_train_baseline_MAK3HTGT/cluster_selection.csv"])

def choose_candidate(vm, cs, primary):
    if {"model", "metric", "value"}.issubset(vm.columns):
        v = vm[vm["metric"].astype(str) == str(primary)].copy()
        v["value"] = pd.to_numeric(v["value"], errors="coerce")
        v = v.dropna(subset=["value"]).sort_values("value", ascending=False)
        if len(v):
            return str(v.iloc[0]["model"])
    if cs is not None and len(cs):
        if {"model", primary}.issubset(cs.columns):
            v = cs[["model", primary]].copy()
            v[primary] = pd.to_numeric(v[primary], errors="coerce")
            v = v.dropna().sort_values(primary, ascending=False)
            if len(v):
                return str(v.iloc[0]["model"])
        if {"model", "selected"}.issubset(cs.columns):
            v = cs[cs["selected"].astype(bool)]
            if len(v):
                return str(v.iloc[0]["model"])
    if {"model", primary}.issubset(vm.columns):
        v = vm[["model", primary]].copy()
        v[primary] = pd.to_numeric(v[primary], errors="coerce")
        v = v.dropna().sort_values(primary, ascending=False)
        if len(v):
            return str(v.iloc[0]["model"])
    raise ValueError(f"Could not find validation-selected candidate for {primary}")

sel_name = choose_candidate(vm, cs, primary)

candidates = {
    "baseline": pickle.load(open(inputs["07_model_train_baseline_MAK3HTGT/model.pkl"], "rb")),
    "alternative": pickle.load(open(inputs["08_model_train_alternative_5AQ82GP1/alternative_model.pkl"], "rb")),
}

def map_candidate(name):
    low = name.lower()
    if low in candidates:
        return low
    if "alt" in low or "08_model_train_alternative" in low:
        return "alternative"
    if "base" in low or "07_model_train_baseline" in low:
        return "baseline"
    for k in candidates:
        if k in low:
            return k
    raise ValueError(f"Unknown candidate {name}; known: {list(candidates)}")

sel_key = map_candidate(sel_name)
selected_model = candidates[sel_key]

test = pd.read_parquet(inputs["test"])
X_test = test[features].copy()

def embedding(model, X):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return model[:-1].transform(X)
    return X

def score_model(model, X):
    labels = model.predict(X)
    Z = embedding(model, X)
    n = len(X)
    k = len(np.unique(labels))
    if k < 2 or k >= n:
        return labels, np.nan, np.nan, np.nan
    sil = silhouette_score(Z, labels, sample_size=min(10000, n), random_state=42)
    ch = calinski_harabasz_score(Z, labels)
    db = davies_bouldin_score(Z, labels)
    return labels, sil, ch, db

rows = []
for name, model in candidates.items():
    _, sil, ch, db = score_model(model, X_test)
    for metric, val in [("silhouette", sil), ("calinski_harabasz", ch), ("davies_bouldin", db)]:
        rows.append({
            "model": name,
            "metric": metric,
            "value": None if pd.isna(val) else float(f"{val:.4g}"),
        })
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test clustering metrics for saved candidates")

sel_row = metrics_df[(metrics_df["model"] == sel_key) & (metrics_df["metric"] == primary)]
if len(sel_row) == 0:
    raise ValueError(f"Primary metric {primary} missing from test metrics")
sel_value = float(sel_row.iloc[0]["value"])
mlca_runtime.record_metric(primary, sel_value, "test")
mlca_runtime.record_fact_candidate("test_score", sel_value, source="test_metrics.csv")

lab_train = selected_model.predict(X_train)
lab_test = selected_model.predict(X_test)
assign = pd.concat([
    pd.DataFrame({"row_id": X_train.index, "partition": "train", "cluster": lab_train}),
    pd.DataFrame({"row_id": X_test.index, "partition": "test", "cluster": lab_test}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")

print(f"selected={sel_key}; features={len(features)}; train={len(X_train)}; test={len(X_test)}; {primary}={sel_value:.4g}")
