import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
fallback_cols = [c for c in train.columns if c not in drop]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

input_cols = None
if hasattr(model, "feature_names_in_"):
    input_cols = list(model.feature_names_in_)
elif isinstance(model, Pipeline):
    for _, step in model.named_steps.items():
        if hasattr(step, "feature_names_in_"):
            input_cols = list(step.feature_names_in_)
            break
if input_cols is None:
    input_cols = fallback_cols

missing = [c for c in input_cols if c not in val.columns]
if missing:
    raise RuntimeError(f"Missing val columns for pipeline: {missing}")

X_val = val[input_cols].copy()
y_raw = val[targets[0]]
if y_raw.dtype == object or str(y_raw.dtype) == "category":
    y_val = pd.factorize(y_raw)[0]
else:
    y_val = np.asarray(y_raw)

scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
result = permutation_importance(
    model, X_val, y_val, scoring=scoring,
    n_repeats=20, random_state=42, n_jobs=1,
)

imp = pd.DataFrame({
    "feature": input_cols,
    "importance_mean": np.round(result.importances_mean, 4),
    "importance_sd": np.round(result.importances_std, 4),
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

top = imp.head(15).copy()
mlca_runtime.save_table(top, "interpretation.csv",
                        caption="Permutation importance (val, 20 repeats, roc_auc)")

top_feature = str(top.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature,
                                   source="interpretation.csv")
mlca_runtime.record_metric("top_importance_mean",
                           float(top.iloc[0]["importance_mean"]), split="val")

print(f"val rows={len(val)} features_used={len(input_cols)}")
print(f"top_feature={top_feature} mean={top.iloc[0]['importance_mean']:.4f}")
