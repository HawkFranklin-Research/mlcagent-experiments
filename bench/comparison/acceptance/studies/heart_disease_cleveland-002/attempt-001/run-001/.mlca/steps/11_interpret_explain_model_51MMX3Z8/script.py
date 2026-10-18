import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

with open(inp["_protocol"], "r", encoding="utf-8") as f:
    proto = json.load(f)

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(obj.keys())}")
    return obj[key]

targets = need(proto, "targets")
excluded = need(proto, "excluded_columns") or []
group = need(proto, "group_column")
task = need(proto, "task")
metric = task.get("primary_metric", "roc_auc")
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
if len(targets) != 1:
    raise ValueError("Expected exactly one target")

df = pd.read_parquet(inp["val"])
target = targets[0]
if target not in df.columns:
    raise KeyError(f"Target {target!r} not found in validation columns")

pipe = None
candidate_keys = [
    "latest_model",
    "07_model_train_logistic_8ET5BK8M/model.pkl",
    "08_model_train_alternative_TE7YJ5NR/alternative_model.pkl",
]
for key in candidate_keys:
    path = inp.get(key)
    if path is None or not path.exists():
        continue
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, Pipeline) and hasattr(obj, "predict_proba") and hasattr(obj, "named_steps"):
        pipe = obj
        break
if pipe is None:
    raise RuntimeError("No complete sklearn Pipeline with predict_proba was found")

if hasattr(pipe, "feature_names_in_"):
    feature_names = [str(c) for c in pipe.feature_names_in_]
else:
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feature_names = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        is_key_like = (
            len(s) > 1
            and s.nunique(dropna=False) == len(s)
            and (
                pd.api.types.is_object_dtype(s.dtype)
                or pd.api.types.is_string_dtype(s.dtype)
                or pd.api.types.is_integer_dtype(s.dtype)
                or pd.api.types.is_bool_dtype(s.dtype)
                or isinstance(s.dtype, pd.CategoricalDtype)
            )
        )
        if not is_key_like:
            feature_names.append(c)

forbidden = set(targets) | set(excluded) | ({group} if group is not None else set())
bad = forbidden.intersection(feature_names)
if bad:
    raise ValueError(f"Forbidden columns present in pipeline inputs: {sorted(bad)}")

missing = [c for c in feature_names if c not in df.columns]
if missing:
    raise KeyError(f"Missing pipeline input columns: {missing}")

X_val = df.loc[:, feature_names].copy()
y_val = df[target]

n_classes = int(task.get("n_classes") or y_val.nunique(dropna=True))
scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"

result = permutation_importance(
    pipe,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

imp = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows, 20 repeats",
)

top_feature = str(imp.iloc[0]["feature"]) if len(imp) else ""
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"validation rows={len(df)}, features={len(feature_names)}, scoring={scoring}")
print(f"top feature={top_feature}")
