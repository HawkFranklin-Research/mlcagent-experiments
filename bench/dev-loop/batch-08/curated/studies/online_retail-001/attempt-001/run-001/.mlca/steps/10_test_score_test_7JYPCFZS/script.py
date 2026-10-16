import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

vm = pd.read_csv(inputs["09_validate_score_validation_89GWDD0X/validation_metrics.csv"])
model_col = next(c for c in vm.columns if "model" in c.lower() or "candidate" in c.lower())
metric_col = next(c for c in vm.columns if "metric" in c.lower())
value_col = next(c for c in vm.columns if "value" in c.lower())
primary = protocol["task"]["primary_metric"]
ascending = primary in ("davies_bouldin",)
sub = vm[vm[metric_col] == primary]
best_row = (sub.sort_values(value_col, ascending=ascending).iloc[0]
            if len(sub) else vm.sort_values(value_col, ascending=ascending).iloc[0])
best_model_name = str(best_row[model_col])

candidates = {
    "baseline": inputs["07_model_train_baseline_SMYBENGX/model.pkl"],
    "alternative": inputs["08_model_train_alternative_602XN080/alternative_model.pkl"],
}
model_path = next((p for k, p in candidates.items() if k.lower() in best_model_name.lower()), None)
if model_path is None:
    model_path = inputs["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Feature column contract: prefer what the saved pipeline was fitted on
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    targets = set(protocol.get("targets", []))
    excluded = set(protocol.get("excluded_columns", []))
    group_col = protocol.get("group_column", None)
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group_col]

X_train = train[feature_cols]
X_test = test[feature_cols]

# Frontend: numeric transformation of features (needed for distance-based metrics)
if isinstance(model, Pipeline) and len(model.steps) >= 2:
    frontend = Pipeline(model.steps[:-1])
    train_labels = np.asarray(model.predict(X_train))
    test_labels = np.asarray(model.predict(X_test))
    X_train_num = np.asarray(frontend.transform(X_train), dtype=float)
    X_test_num = np.asarray(frontend.transform(X_test), dtype=float)
else:
    with open(inputs["05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_retain_features_YJWZ54M0/selector_all.pkl"], "rb") as f:
        sel = pickle.load(f)
    X_train_num = np.asarray(sel.transform(pre.transform(X_train)), dtype=float)
    X_test_num = np.asarray(sel.transform(pre.transform(X_test)), dtype=float)
    train_labels = np.asarray(model.predict(X_train_num))
    test_labels = np.asarray(model.predict(X_test_num))

sample_n = min(10000, len(X_test_num))
sil = float(silhouette_score(X_test_num, test_labels, sample_size=sample_n, random_state=42))
ch = float(calinski_harabasz_score(X_test_num, test_labels))
db = float(davies_bouldin_score(X_test_num, test_labels))

metrics_df = pd.DataFrame({
    "model": [best_model_name, best_model_name, best_model_name],
    "metric": ["silhouette", "calinski_harabasz", "davies_bouldin"],
    "value": [round(sil, 6), round(ch, 6), round(db, 6)],
})
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test clustering metrics")

mlca_runtime.record_metric(primary, sil, "test")
mlca_runtime.record_fact_candidate("test_score", sil, source="test_metrics.csv")

assignments = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": train_labels}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": test_labels}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, "assignments.parquet")

print(f"Model: {best_model_name}")
print(f"Test {primary}: {sil:.4f}")
print(f"Assignments: {len(assignments)}")
