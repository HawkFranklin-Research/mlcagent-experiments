import mlca_runtime
import json
import pickle
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

test = pd.read_parquet(inputs["test"])

with open(inputs["05_preprocess_fit_preprocessing_CQ5BZKTY/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_XJNGW7H2/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in test.columns if c not in drop]
X_test = test[feature_cols].copy()

if hasattr(model, "named_steps"):
    pred = model.predict(X_test)
else:
    if hasattr(selector, "named_steps"):
        X_test = selector.transform(X_test)
    else:
        X_test = preprocessor.transform(X_test)
        X_test = selector.transform(X_test)
    pred = model.predict(X_test)

out = pd.DataFrame({
    "row_id": test.index,
    "outcome": test[targets[0]],
    "prediction": pred
})
mlca_runtime.save_row_outputs(out, "test_predictions.csv")

print(f"test rows: {len(test)}")
print(f"features: {len(feature_cols)}")
print(f"model: {inputs['latest_model'].name}")
