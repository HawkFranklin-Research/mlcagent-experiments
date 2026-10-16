import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_F5B2KN9D/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_Y4ZJEG70/selector.pkl"])
mod = load_pkl(inputs["07_model_train_logistic_CV1F3546/model.pkl"])
alt = load_pkl(inputs["08_model_train_alternative_Z4FMHBXH/alternative_model.pkl"])
latest = load_pkl(inputs["latest_model"])

candidates = [latest, alt, mod, sel, pre]
pipeline = None
for obj in candidates:
    if isinstance(obj, Pipeline) and hasattr(obj, "feature_names_in_"):
        if hasattr(obj, "predict_proba") or hasattr(obj, "decision_function"):
            pipeline = obj
            break

if pipeline is None:
    raise RuntimeError("No complete pipeline found among saved objects")

expected_features = pipeline.feature_names_in_

val = pd.read_parquet(inputs["val"])
X_val = val[expected_features].copy()

target_col = targets[0]
y_val = val[target_col]

classes = pipeline.classes_
if classes.dtype == "object":
    y_val = (y_val == classes[1]).astype(int)
else:
    if y_val.dtype == "object":
        y_val = y_val.str.contains(">50K").astype(int)

scoring = "roc_auc" if task["type"] == "binary" else "roc_auc_ovr"
result = permutation_importance(
    pipeline, X_val, y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

imp_df = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(imp_df, "interpretation.csv", caption="Permutation importance top 15 features")

top_feature = imp_df.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Top feature: {top_feature}")
print(f"Validation rows: {len(X_val)}")
