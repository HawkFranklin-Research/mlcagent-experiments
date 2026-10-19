import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "task"]:
    if key not in protocol:
        raise KeyError(f"missing {key}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
task = protocol["task"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
scoring = task["primary_metric"]

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

def is_key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=True) == s.notna().sum()
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == s.notna().sum()
    return False

drop = set(targets) | set(excluded) | {group}
feature_cols = [
    c for c in train_df.columns
    if c not in drop and not is_key_like(train_df[c])
]
feature_set = set(feature_cols)

def load_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            import cloudpickle
            f.seek(0)
            return cloudpickle.load(f)

candidate_keys = [
    "latest_model",
    "07_model_train_logistic_G7CT94CC/model.pkl",
    "08_model_train_alternative_PSET1F82/alternative_model.pkl",
]
candidates = []
for key in candidate_keys:
    obj = load_pkl(inputs[key])
    if not hasattr(obj, "predict_proba"):
        continue
    names = list(getattr(obj, "feature_names_in_", []))
    if names and set(names) == feature_set:
        candidates.append((key, obj, names))

if not candidates:
    raise RuntimeError("No complete frozen pipeline accepts the declared feature columns")

def rank(c):
    key, obj, _ = c
    nsteps = len(getattr(obj, "named_steps", {}))
    pref = 2 if key == "latest_model" else (1 if key.startswith("07_") else 0)
    return (nsteps, pref)

chosen_key, pipeline, names = max(candidates, key=rank)
X_val = val_df.loc[:, names].copy()
y_val = val_df[targets[0]]

result = permutation_importance(
    pipeline,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

importance = pd.DataFrame({
    "feature": list(names),
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

for col in ["importance_mean", "importance_sd"]:
    importance[col] = importance[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(importance, "interpretation.csv", caption="Top permutation importance features on validation rows")
top_feature = str(importance.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"chosen={chosen_key} val_rows={len(X_val)} features={len(feature_cols)}")
print(f"top_feature={top_feature}")
