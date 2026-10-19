import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
    metric = task["primary_metric"]
except KeyError:
    print("Available top-level protocol keys:", list(protocol.keys()))
    raise
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and s.notna().all()
    if pd.api.types.is_integer_dtype(s):
        diffs = s.sort_values().diff().dropna()
        return (
            s.nunique(dropna=True) == len(s)
            and s.is_monotonic_increasing
            and len(diffs) > 0
            and (diffs == 1).all()
        )
    return False

base_cols = [c for c in train.columns if c not in drop]
feature_cols = [c for c in base_cols if not is_key_like(train[c])]
target_col = targets[0]
y_val = val[target_col]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

obj_keys = [
    "05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl",
    "06_features_select_features_F6NPFNPS/selector.pkl",
    "07_model_train_logistic_Q9WJ6BHH/model.pkl",
    "latest_model",
    "08_model_train_alternative_3WAJAX78/alternative_model.pkl",
]
objs = {k: load_pkl(inputs[k]) for k in obj_keys if k in inputs}

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        first = obj.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def accepts_raw(obj):
    names = input_names(obj)
    return hasattr(obj, "predict_proba") and names is not None and set(names) == set(feature_cols) and len(names) == len(feature_cols)

complete = None
for k in ["latest_model", "07_model_train_logistic_Q9WJ6BHH/model.pkl", "08_model_train_alternative_3WAJAX78/alternative_model.pkl"]:
    if k in objs and accepts_raw(objs[k]):
        complete = objs[k]
        break

if complete is None:
    pre = objs.get("05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl")
    sel = objs.get("06_features_select_features_F6NPFNPS/selector.pkl")
    mod = objs.get("07_model_train_logistic_Q9WJ6BHH/model.pkl")
    if mod is not None and accepts_raw(mod):
        complete = mod
    elif sel is not None:
        sel_names = input_names(sel)
        if sel_names is not None and set(sel_names) == set(feature_cols):
            complete = Pipeline([("selector", sel), ("model", mod)])
        elif pre is not None:
            complete = Pipeline([("preprocessor", pre), ("selector", sel), ("model", mod)])
        else:
            complete = Pipeline([("selector", sel), ("model", mod)])
    elif pre is not None:
        complete = Pipeline([("preprocessor", pre), ("model", mod)])
    else:
        complete = mod

if complete is None:
    raise RuntimeError("No complete frozen pipeline could be loaded.")

names = input_names(complete)
if names is not None:
    missing = [c for c in names if c not in val.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    X_val = val[names].copy()
    imp_features = names
else:
    X_val = val[feature_cols].copy()
    imp_features = feature_cols

result = permutation_importance(
    complete,
    X_val,
    y_val,
    scoring=metric,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

def sig4(x):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.4g}")

out = pd.DataFrame(
    {
        "feature": imp_features,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

out["importance_mean"] = out["importance_mean"].map(sig4)
out["importance_sd"] = out["importance_sd"].map(sig4)

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance on validation rows")
top_feature = out.iloc[0]["feature"] if len(out) else None
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print("features", len(imp_features), "validation_rows", len(X_val))
print("top_feature", top_feature)
