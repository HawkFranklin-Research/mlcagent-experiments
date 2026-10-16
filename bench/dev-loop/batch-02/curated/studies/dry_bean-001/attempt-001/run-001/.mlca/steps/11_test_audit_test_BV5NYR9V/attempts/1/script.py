import mlca_runtime
import json
import pickle
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing key '{k}' in _protocol. Available: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = req(protocol, "group_column")
task = req(protocol, "task")

test = pd.read_parquet(inputs["test"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
elif hasattr(model, "named_steps"):
    feature_cols = None
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            feature_cols = list(step.feature_names_in_)
            break
    if feature_cols is None:
        drop = set(targets) | set(excluded)
        if group is not None:
            drop.add(group)
        feature_cols = [c for c in test.columns if c not in drop]
else:
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feature_cols = [c for c in test.columns if c not in drop]

X_test = test[feature_cols].copy()
y_test = test[targets[0]].copy()
pred = model.predict(X_test)

out = pd.DataFrame({"outcome": y_test, "prediction": pred}, index=test.index)
mlca_runtime.save_row_outputs(out, "test_predictions.csv")

print(f"test rows: {len(test)}")
print(f"features: {len(feature_cols)}")
print(f"model: {inputs['latest_model'].name}")
