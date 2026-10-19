import json
import pickle

import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train rows={len(train)} val rows={len(val)}")

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)
candidates = [c for c in train.columns if c not in drop]

def is_key_like(s: pd.Series) -> bool:
    if s.isna().any() or s.nunique(dropna=False) != len(s):
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )

features = [c for c in candidates if not is_key_like(train[c])]
missing = [c for c in features if c not in val.columns]
if missing:
    raise KeyError(f"Validation partition is missing feature columns: {missing}")
print(f"features={len(features)}")

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict) and "model" in model:
    model = model["model"]

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    missing_model = [c for c in model_features if c not in features]
    if missing_model:
        raise ValueError(f"Model expects columns outside feature list: {missing_model}")
    features = model_features
elif not hasattr(model, "predict_proba"):
    raise TypeError("Loaded model does not expose predict_proba")

X_val = val[features]
y_val = val[targets[0]]
print(f"validation feature matrix={X_val.shape}")

def scorer(est, X, y):
    proba = est.predict_proba(X)
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring=scorer,
    n_repeats=20,
    random_state=SEED,
    n_jobs=1,
)

importance = pd.DataFrame({
    "feature": features,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

def sig4(x):
    if pd.isna(x):
        return np.nan
    return 0.0 if x == 0 else float(f"{x:.4g}")

importance["importance_mean"] = importance["importance_mean"].map(sig4)
importance["importance_sd"] = importance["importance_sd"].map(sig4)

mlca_runtime.save_table(
    importance,
    "interpretation.csv",
    caption="Permutation importance on validation rows",
)
top_feature = importance.loc[0, "feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"top_feature={top_feature}")
