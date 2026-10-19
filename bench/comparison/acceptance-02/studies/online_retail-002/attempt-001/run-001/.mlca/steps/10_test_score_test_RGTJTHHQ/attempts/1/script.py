import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
primary = protocol["task"]["primary_metric"]
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train_df = pd.read_parquet(inputs["train"])
test_df = pd.read_parquet(inputs["test"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = []
for c in train_df.columns:
    if c in drop:
        continue
    s = train_df[c]
    name = str(c).lower()
    if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=True) == len(train_df):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == len(train_df) and any(t in name for t in ["id", "key", "code", "index", "uuid"]):
        continue
    feature_cols.append(c)

try:
    vm = pd.read_parquet(inputs["09_validate_score_validation_ERHSD5DA/validation_metrics.parquet"])
except Exception:
    vm = pd.read_csv(inputs["09_validate_score_validation_ERHSD5DA/validation_metrics.csv"])

def direction(metric):
    m = str(metric).lower()
    return "min" if "davies" in m or "bouldin" in m else "max"

def select_model(vm, metric):
    if {"model", "metric", "value"}.issubset(set(vm.columns)):
        sub = vm[vm["metric"].astype(str).str.lower() == str(metric).lower()]
        if len(sub):
            scores = sub.groupby("model")["value"].mean()
            return str(scores.idxmax() if direction(metric) == "max" else scores.idxmin())
    model_col = "model" if "model" in vm.columns else vm.columns[0]
    for c in vm.columns:
        if str(c).lower() == str(metric).lower():
            scores = vm.set_index(model_col)[c]
            return str(scores.idxmax() if direction(metric) == "max" else scores.idxmin())
    raise ValueError(f"{metric} not found in validation metrics")

candidates = {}
with open(inputs["07_model_train_baseline_M82AMBVM/model.pkl"], "rb") as f:
    candidates["baseline"] = pickle.load(f)
with open(inputs["08_model_train_alternative_DY0NT3JN/alternative_model.pkl"], "rb") as f:
    candidates["alternative"] = pickle.load(f)

selected = select_model(vm, primary)
sel_l = selected.lower()
if "baseline" in sel_l or "base" in sel_l:
    selected_key = "baseline"
elif "alt" in sel_l:
    selected_key = "alternative"
elif selected in candidates:
    selected_key = selected
else:
    selected_key = next((k for k in candidates if k in sel_l or sel_l in k), None)
    if selected_key is None:
        selected_key = "alternative" if "alt" in sel_l else list(candidates)[0]

frozen = candidates[selected_key]

def expected_columns(model):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                break
    return list(cols) if cols is not None else feature_cols

def transform_features(model, X):
    if hasattr(model, "steps") and len(model.steps) > 1:
        Z = model[:-1].transform(X)
    else:
        Z = X
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z)

def predict_labels(model, X):
    if not hasattr(model, "predict"):
        raise AttributeError("frozen model has no predict")
    return model.predict(X)

expected = expected_columns(frozen)
X_test = test_df[expected]
labels_test = predict_labels(frozen, X_test)
Z_test = transform_features(frozen, X_test)
sample_size = min(10000, len(Z_test))
vals = {
    "silhouette": float(silhouette_score(Z_test, labels_test, sample_size=sample_size, random_state=42)),
    "calinski_harabasz": float(calinski_harabasz_score(Z_test, labels_test)),
    "davies_bouldin": float(davies_bouldin_score(Z_test, labels_test)),
}

metrics_df = pd.DataFrame([
    {"model": selected, "metric": k, "value": float(f"{v:.4g}")}
    for k, v in vals.items()
])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", "Test clustering metrics")

def assignments(model, df, partition):
    X = df[expected]
    return pd.DataFrame({
        "row_id": df.index,
        "partition": partition,
        "cluster": predict_labels(model, X),
    })

assign_df = pd.concat([
    assignments(frozen, train_df, "train"),
    assignments(frozen, test_df, "test"),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign_df, "assignments.parquet")

metric_key = {
    "silhouette": "silhouette",
    "calinski-harabasz": "calinski_harabasz",
    "calinski_harabasz": "calinski_harabasz",
    "davies-bouldin": "davies_bouldin",
    "davies_bouldin": "davies_bouldin",
}.get(str(primary).lower(), str(primary).lower())
primary_val = vals[metric_key]
mlca_runtime.record_metric(primary, primary_val, split="test")
mlca_runtime.record_fact_candidate("test_score", primary_val, source="test_metrics.csv")

print(f"frozen={selected} train_rows={len(train_df)} test_rows={len(test_df)}")
print(f"test_{primary}={primary_val:.4f}")
