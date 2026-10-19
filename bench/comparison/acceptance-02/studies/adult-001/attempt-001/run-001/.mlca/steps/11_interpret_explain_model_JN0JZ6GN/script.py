import json
import numpy as np
import pandas as pd
import pickle
import mlca_runtime
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task.get("primary_metric") or ("roc_auc" if task.get("type") == "binary" else "roc_auc_ovr")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
feature_cols = []
for c in train.columns:
    if c in exclude:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith(("string", "category")):
        if len(s) > 0 and s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if len(s) > 0 and s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    perm_cols = [c for c in list(model.feature_names_in_) if c in val.columns]
else:
    perm_cols = feature_cols

X_val = val[perm_cols].copy()
y_raw = val[targets[0]]
if hasattr(model, "classes_") and hasattr(model.classes_, "__len__") and len(model.classes_) == 2:
    y_val = (y_raw == model.classes_[1]).astype(int)
else:
    y_val = y_raw

def scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))

result = permutation_importance(model, X_val, y_val, scoring=scorer,
                                n_repeats=20, random_state=42, n_jobs=1)

imp = pd.DataFrame({
    "feature": perm_cols,
    "importance_mean": np.round(result.importances_mean, 4),
    "importance_sd": np.round(result.importances_std, 4),
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(imp, "interpretation.csv",
                        caption="Permutation importance on validation (ROC AUC, 20 repeats), top 15 features")
top_feature = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"features={len(perm_cols)} val_rows={len(X_val)} top={top_feature} mean={imp.iloc[0]['importance_mean']}")
