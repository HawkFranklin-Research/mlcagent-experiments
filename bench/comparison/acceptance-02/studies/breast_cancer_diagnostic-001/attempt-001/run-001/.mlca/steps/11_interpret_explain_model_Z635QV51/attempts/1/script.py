import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
SEED = 42
N_REPEATS = 20
TOP_N = 15

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
    primary_metric = task["primary_metric"]
except KeyError as exc:
    raise ValueError(
        f"Missing protocol key {exc!r}; available top-level keys: {sorted(protocol.keys())}"
    ) from exc

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

target_col = targets[0]
y_val = val[target_col]

banned = set(targets) | set(excluded)
if group_col:
    banned.add(group_col)


def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False


feature_cols = [
    c for c in train.columns
    if c not in banned and not is_key_like(train[c])
]

candidates = [
    "latest_model",
    "07_model_train_logistic_94MG7111/model.pkl",
    "08_model_train_alternative_227Y318V/alternative_model.pkl",
]
pipe = None
for key in candidates:
    with open(inputs[key], "rb") as handle:
        obj = pickle.load(handle)
    if hasattr(obj, "named_steps") and hasattr(obj, "predict_proba"):
        pipe = obj
        break
if pipe is None:
    raise ValueError("No complete probabilistic pipeline found in saved model candidates")

if hasattr(pipe, "feature_names_in_"):
    expected = list(pipe.feature_names_in_)
    if set(expected) & banned:
        raise ValueError("Saved pipeline expects a target, excluded, or group column")
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise ValueError(f"Saved pipeline expects missing columns: {missing}")
    feature_cols = expected

X_val = val[feature_cols]

n_classes = len(getattr(pipe, "classes_", [])) or int(y_val.nunique())
if n_classes == 2:
    scoring = "roc_auc"
else:
    scoring = "roc_auc_ovr"

result = permutation_importance(
    pipe,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=N_REPEATS,
    random_state=SEED,
    n_jobs=1,
)

table = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(TOP_N).reset_index(drop=True)


def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")


table["importance_mean"] = table["importance_mean"].map(sig4)
table["importance_sd"] = table["importance_sd"].map(sig4)

mlca_runtime.save_table(
    table,
    "interpretation.csv",
    caption="Validation permutation importance using the frozen complete pipeline",
)

top_feature = str(table.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"features={len(feature_cols)} val_rows={len(X_val)} repeats={N_REPEATS}")
print(f"scoring={scoring} top_feature={top_feature}")
