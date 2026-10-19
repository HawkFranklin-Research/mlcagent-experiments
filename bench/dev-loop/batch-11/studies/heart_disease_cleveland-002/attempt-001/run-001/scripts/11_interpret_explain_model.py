import json
import pickle

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r} in protocol; available keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]
task = protocol["task"]
metric = task.get("primary_metric", "roc_auc")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

nrow = len(train)
base_cols = [c for c in train.columns if c not in set(targets) | excluded and c != group_col]
features = []
for col in base_cols:
    s = train[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == nrow:
            continue
    elif pd.api.types.is_integer_dtype(s):
        name = str(col).lower()
        if s.nunique(dropna=False) == nrow and any(tok in name for tok in ("id", "code", "key", "index", "row", "seq")):
            continue
    features.append(col)

with inp["latest_model"].open("rb") as f:
    est = pickle.load(f)
if not hasattr(est, "predict_proba"):
    raise TypeError("Latest model does not provide predict_proba")

cols = list(getattr(est, "feature_names_in_", features))
missing = [c for c in cols if c not in val.columns]
if missing:
    raise KeyError(f"Validation data is missing model input columns: {missing}")

X_val = val[cols].copy()
y_val = val[targets[0]]

n_classes = int(task.get("n_classes", y_val.nunique()))
if n_classes == 2 or metric in ("roc_auc", "auc"):
    scoring = "roc_auc"
else:
    scoring = "roc_auc_ovr"

res = permutation_importance(
    est,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

imp = pd.DataFrame({
    "feature": np.asarray(X_val.columns),
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ("importance_mean", "importance_sd"):
    imp[col] = imp[col].map(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows using the frozen pipeline.",
)
top_feature = str(imp.loc[0, "feature"]) if len(imp) else ""
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows={len(train)}, val rows={len(val)}")
print(f"features used={len(features)}, model inputs={len(cols)}")
print(f"top feature={top_feature}")
