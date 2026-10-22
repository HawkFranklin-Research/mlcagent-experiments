import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import mlca_runtime

mlca_runtime.set_seed(42)
seed = 42
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = set(protocol["excluded_columns"])
    group = protocol["group_column"]
    task = protocol["task"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {list(protocol)}") from exc

val = pd.read_parquet(paths["val"])
target = targets[0]

def is_keylike(s: pd.Series) -> bool:
    n = int(s.notna().sum())
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return int(s.nunique(dropna=True)) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return int(s.nunique(dropna=True)) == n and n > 1 and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

feature_cols = [
    c for c in val.columns
    if c not in set(targets) and c not in excluded and c != group and not is_keylike(val[c])
]

def input_names(est):
    names = getattr(est, "feature_names_in_", None)
    if names is None and isinstance(est, Pipeline):
        for _, step in est.steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    return None if names is None else list(names)

candidate_keys = [
    "latest_model",
    "08_model_train_logistic_MMDVH86M/model.pkl",
    "09_model_train_alternative_4GY8FEXD/alternative_model.pkl",
]
model = None
model_key = None
model_input_cols = None
seen = set()
for key in candidate_keys:
    path = paths[key]
    if str(path) in seen:
        continue
    seen.add(str(path))
    obj = pickle.loads(path.read_bytes())
    if not hasattr(obj, "predict_proba"):
        continue
    names = input_names(obj)
    if names is not None and set(names) == set(feature_cols):
        model = obj
        model_key = key
        model_input_cols = names
        break

if model is None:
    raise RuntimeError(f"No complete saved model accepts the raw feature columns: {feature_cols}")

X_val = val[model_input_cols]
y_raw = val[target]
classes = getattr(model, "classes_", None)
if classes is not None and len(classes) == 2:
    y_val = (y_raw == classes[1]).astype(int)
else:
    y_val = pd.factorize(y_raw)[0]

metric = task.get("primary_metric")
n_classes = int(task.get("n_classes", 2))
if metric in (None, "", "roc_auc", "auc"):
    scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
elif metric in ("roc_auc_ovr", "roc_auc_ovo"):
    scoring = metric
else:
    scoring = "roc_auc"

result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=seed,
    n_jobs=1,
)

interp = pd.DataFrame({
    "feature": model_input_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15)
interp[["importance_mean", "importance_sd"]] = interp[["importance_mean", "importance_sd"]].round(4)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation importance, top 15 features")

top_feature = str(interp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"val rows: {len(val)}")
print(f"features: {len(feature_cols)}")
print(f"model: {model_key}")
print(f"top feature: {top_feature}")
print(f"top importance mean: {interp.iloc[0]['importance_mean']}")
