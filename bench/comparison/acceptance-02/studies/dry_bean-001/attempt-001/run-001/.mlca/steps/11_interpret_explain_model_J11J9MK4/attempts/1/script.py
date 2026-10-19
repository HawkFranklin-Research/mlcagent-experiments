import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task key primary_metric; available: {list(task)}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return s.nunique(dropna=False) == len(s) and ("id" in name or "key" in name or name in {"index", "row"})
    return False

excluded_set = set(targets) | set(excluded)
if group_col is not None:
    excluded_set.add(group_col)
feature_cols = [c for c in train.columns if c not in excluded_set and not key_like(train[c])]

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)
if not isinstance(model, Pipeline):
    raise TypeError("Saved model is not a complete sklearn Pipeline")
if hasattr(model, "feature_names_in_"):
    model_cols = list(model.feature_names_in_)
    if set(model_cols) != set(feature_cols):
        raise ValueError("Model input columns do not match declared feature columns")
    feature_cols = model_cols

if len(targets) != 1:
    raise ValueError("This interpretation script expects exactly one target column")
Xv = val[feature_cols]
y = val[targets[0]]
n_classes = task.get("n_classes", int(train[targets[0]].nunique()))
supported = {"roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"}
scoring = primary_metric if primary_metric in supported else ("roc_auc_ovr" if n_classes > 2 else "roc_auc")

result = permutation_importance(
    model, Xv, y, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)
imp = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ["importance_mean", "importance_sd"]:
    imp[col] = imp[col].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    imp, "interpretation.csv",
    caption=f"Validation permutation importance, scoring={scoring}, n_repeats=20."
)
top_feature = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"val_rows={len(val)} features={len(feature_cols)}")
print(f"top_feature={top_feature}")
