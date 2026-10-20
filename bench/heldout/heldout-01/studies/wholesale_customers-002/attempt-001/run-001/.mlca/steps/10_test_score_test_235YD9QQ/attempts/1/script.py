import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    primary_metric = protocol["task"]["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing task.primary_metric; available top-level keys: {list(protocol)}") from exc

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
test = pd.read_parquet(inputs["test"])

targets = list(protocol.get("targets", [])) + list(protocol.get("task", {}).get("targets", []))
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)
drop_cols = set(targets + excluded + ([group] if group else []))
candidate_cols = [c for c in train.columns if c not in drop_cols]

def key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        if n == 1:
            return True
        d = s.sort_values().diff().dropna()
        return bool((d == 1).all())
    return False

feature_columns = [c for c in candidate_cols if not key_like(train[c])]
if not feature_columns:
    raise ValueError("No feature columns remain after exclusions and key-like detection")

def model_features(model):
    return list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_columns

def predict_labels(model, df):
    feats = model_features(model)
    return model.predict(df.loc[:, feats])

def metric_space(model, X):
    if hasattr(model, "named_steps") and len(model.named_steps) > 1:
        try:
            return model[:-1].transform(X)
        except Exception:
            pass
    if hasattr(model, "transform"):
        try:
            return model.transform(X)
        except Exception:
            pass
    return X

def score_model(model, df):
    feats = model_features(model)
    X = df.loc[:, feats]
    labels = model.predict(X)
    Z = metric_space(model, X)
    vals = {m: float("nan") for m in ["silhouette", "calinski_harabasz", "davies_bouldin"]}
    k = len(set(labels))
    if k > 1 and len(labels) > k:
        sample_size = min(10000, len(Z)) if len(Z) > 10000 else None
        for name, fn in [
            ("silhouette", lambda: silhouette_score(Z, labels, sample_size=sample_size, random_state=42)),
            ("calinski_harabasz", lambda: calinski_harabasz_score(Z, labels)),
            ("davies_bouldin", lambda: davies_bouldin_score(Z, labels)),
        ]:
            try:
                vals[name] = float(fn())
            except Exception:
                pass
    return labels, vals

with open(inputs["07_model_train_baseline_9CYH36JK/model.pkl"], "rb") as f:
    baseline = pickle.load(f)
with open(inputs["08_model_train_alternative_R5ZEQMH2/alternative_model.pkl"], "rb") as f:
    alternative = pickle.load(f)
models = {"baseline": baseline, "alternative": alternative}

val_scores = {}
for name, model in models.items():
    _, vals = score_model(model, val)
    val_scores[name] = vals.get(primary_metric, float("nan"))

valid = [n for n in models if np.isfinite(val_scores[n])]
if valid:
    lower_better = primary_metric == "davies_bouldin"
    selected_name = (min if lower_better else max)(valid, key=lambda n: val_scores[n])
else:
    selected_name = "baseline"
selected_model = models[selected_name]

test_labels, test_vals = score_model(selected_model, test)
primary_value = test_vals.get(primary_metric, float("nan"))

metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
metrics_df = pd.DataFrame({
    "model": selected_name,
    "metric": metrics,
    "value": [float(f"{test_vals[m]:.4g}") if np.isfinite(test_vals[m]) else np.nan for m in metrics],
})
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test clustering metrics for selected candidate")
mlca_runtime.record_metric(primary_metric, primary_value, "test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")

train_labels = predict_labels(selected_model, train)
assign_df = pd.concat(
    [
        pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": train_labels}),
        pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": test_labels}),
    ],
    ignore_index=True,
)
mlca_runtime.save_row_outputs(assign_df, "assignments.parquet")

print(f"selected={selected_name} primary={primary_metric} test={primary_value:.4g}")
print(f"train={len(train)} test={len(test)}")
