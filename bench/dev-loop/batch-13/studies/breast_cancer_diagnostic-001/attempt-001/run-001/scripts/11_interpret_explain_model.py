import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key}; available keys: {list(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto["group_column"]
task = proto["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and s.is_unique:
        return True
    return False

protected = set(targets) | set(excluded)
if group_col is not None:
    protected.add(group_col)
feature_cols = [c for c in train.columns if c not in protected and not is_key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after applying protocol exclusions")

candidate_paths = [
    inp["07_model_train_logistic_QHAMT8ZK/model.pkl"],
    inp["08_model_train_alternative_SCMRPCK3/alternative_model.pkl"],
    inp["latest_model"],
]
candidates = []
for path in candidate_paths:
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, Pipeline) and hasattr(obj, "predict_proba"):
        names = list(getattr(obj, "feature_names_in_", feature_cols))
        if set(names) & protected:
            continue
        candidates.append((len(obj.named_steps), obj, names))
if not candidates:
    raise RuntimeError("No saved complete pipeline with predict_proba was found")
candidates.sort(key=lambda t: t[0], reverse=True)
_, pipe, names = candidates[0]
missing = set(names) - set(train.columns)
if missing:
    raise KeyError(f"Pipeline expects missing columns: {sorted(missing)}")

X_val = val[names].copy()
y_val = val[targets[0]].copy()
n_classes = len(pd.unique(y_val.dropna()))
if n_classes == 2:
    classes = getattr(pipe, "classes_", np.sort(pd.unique(y_val)))
    pos = classes[1]
    y_eval = (y_val == pos).astype(int).to_numpy()
    scoring = "roc_auc"
else:
    y_eval = y_val.to_numpy()
    scoring = "roc_auc_ovr"

result = permutation_importance(
    pipe, X_val, y_eval, scoring=scoring, n_repeats=20, random_state=SEED, n_jobs=1
)
interp = pd.DataFrame({
    "feature": names,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ["importance_mean", "importance_sd"]:
    interp[col] = interp[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation importance, 20 repeats")
top_feature = str(interp.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"train rows {len(train)}, val rows {len(val)}")
print(f"features {len(names)}, task metric {metric}, scoring {scoring}")
print(f"top feature {top_feature}, mean {interp.loc[0, 'importance_mean']}")
print(f"saved interpretation.csv rows {len(interp)}")
