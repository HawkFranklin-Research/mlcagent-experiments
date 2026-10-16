import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

targets = set(protocol.get("targets", []))
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column", None)

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if train[c].dtype == object and train[c].nunique() == len(train):
        continue
    if pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train):
        continue
    feature_cols.append(c)

X_train = train[feature_cols]
X_test = test[feature_cols]

vm = pd.read_csv(inputs["09_validate_score_validation_89GWDD0X/validation_metrics.csv"])
model_col = next(c for c in vm.columns if "model" in c.lower() or "candidate" in c.lower())
metric_col = next(c for c in vm.columns if "metric" in c.lower())
value_col = next(c for c in vm.columns if "value" in c.lower())

primary = protocol["task"]["primary_metric"]
best_row = vm[vm[metric_col] == primary].sort_values(value_col, ascending=False).iloc[0]
best_model_name = best_row[model_col]

candidates = {
    "baseline": inputs["07_model_train_baseline_SMYBENGX/model.pkl"],
    "alternative": inputs["08_model_train_alternative_602XN080/alternative_model.pkl"],
}
model_path = next((p for k, p in candidates.items() if k.lower() in best_model_name.lower()), None)
if model_path is None:
    model_path = inputs["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

train_labels = model.predict(X_train)
test_labels = model.predict(X_test)

sample_n = min(10000, len(X_test))
sil = silhouette_score(X_test, test_labels, sample_size=sample_n, random_state=42)
ch = calinski_harabasz_score(X_test, test_labels)
db = davies_bouldin_score(X_test, test_labels)

metrics_df = pd.DataFrame({
    "model": [best_model_name] * 3,
    "metric": ["silhouette", "calinski_harabasz", "davies_bouldin"],
    "value": [sil, ch, db],
})
mlca_runtime.save_table(metrics_df, "test_metrics.csv", "Test set clustering metrics")

mlca_runtime.record_metric(primary, float(sil), "test")
mlca_runtime.record_fact_candidate("test_score", float(sil), source="test_metrics.csv")

assignments = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": train_labels}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": test_labels}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, "assignments.parquet")

print(f"Selected model: {best_model_name}")
print(f"Test silhouette: {sil:.4f}")
print(f"Assignments rows: {len(assignments)}")
