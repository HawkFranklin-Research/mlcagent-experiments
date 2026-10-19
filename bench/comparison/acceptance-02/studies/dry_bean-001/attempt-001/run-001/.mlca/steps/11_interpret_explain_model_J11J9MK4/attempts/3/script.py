import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
n_classes = int(train[targets[0]].nunique())

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        n = str(s.name).lower()
        return s.nunique(dropna=False) == len(s) and ("id" in n or "key" in n or n in {"index", "row"})
    return False

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)
if not isinstance(model, Pipeline):
    raise TypeError("Latest model is not a Pipeline")
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)

Xv = val[feature_cols]
y = val[targets[0]]

supported = {"roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"}
scoring = primary_metric if primary_metric in supported else ("roc_auc_ovr" if n_classes > 2 else "roc_auc")

result = permutation_importance(
    model, Xv, y, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].round(4)

mlca_runtime.save_table(
    imp, "interpretation.csv",
    caption=f"Validation permutation importance, scoring={scoring}, n_repeats=20."
)
top = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"val_rows={len(val)} features={len(feature_cols)} scoring={scoring}")
print(f"top_feature={top}")
