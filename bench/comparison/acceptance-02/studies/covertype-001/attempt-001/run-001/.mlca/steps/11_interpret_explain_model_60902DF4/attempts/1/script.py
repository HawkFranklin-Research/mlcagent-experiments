import json
import pickle
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
metric = task["primary_metric"]
group_col = protocol.get("group_column")
drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=True) == n and s.is_monotonic_increasing
    return False

feature_cols = [c for c in train.columns if c not in drop_cols and not key_like(train[c])]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

candidates = [
    inp["latest_model"],
    inp["07_model_train_logistic_Z4Z606HV/model.pkl"],
    inp["08_model_train_alternative_JKAAJDG0/alternative_model.pkl"],
]
model = None
for path in candidates:
    obj = load_pickle(path)
    if isinstance(obj, Pipeline) and hasattr(obj, "predict_proba") and hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        if set(feature_cols).issubset(set(names)):
            model = obj
            break
if model is None:
    raise RuntimeError("No complete frozen Pipeline with predict_proba and raw feature names was found.")

model_names = list(model.feature_names_in_)
bad = set(model_names) & drop_cols
if bad:
    raise RuntimeError(f"Model expects excluded/target/group columns: {sorted(bad)}")
if set(model_names) != set(feature_cols):
    raise RuntimeError("Model feature_names_in_ does not match the frozen feature schema.")

feature_cols = model_names
X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring=metric,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

imp = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
imp["importance_mean"] = imp["importance_mean"].round(4)
imp["importance_sd"] = imp["importance_sd"].round(4)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Validation permutation importance")
top_feature = str(imp.iloc[0]["feature"]) if len(imp) else ""
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"validation rows: {len(val)}; model features: {len(feature_cols)}; repeats: 20")
print(f"top feature: {top_feature}")
