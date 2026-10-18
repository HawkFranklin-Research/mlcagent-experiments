import mlca_runtime as mlca
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline

mlca.set_seed(42)
inputs = mlca.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
task = need(proto, "task")
metric = task.get("primary_metric")
if metric is None:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task.keys())}")
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(col, s):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    name = str(col).lower()
    if any(tok in name for tok in ("id", "index", "row", "key")):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = np.sort(s.dropna().unique())
        if len(vals) > 1:
            diffs = np.diff(vals)
            if np.all(diffs == diffs[0]) and diffs[0] in (1, -1):
                return True
    return False

protected = set(targets) | set(excluded)
if group is not None:
    protected.add(group)
candidates = [c for c in train.columns if c not in protected]
features = [c for c in candidates if not is_key_like(c, train[c])]

model_paths = [
    "07_model_train_logistic_D2WYGTBM/model.pkl",
    "latest_model",
    "08_model_train_alternative_5QQJK8QV/alternative_model.pkl",
]
best_model = None
best_steps = -1
for key in model_paths:
    with open(inputs[key], "rb") as fh:
        obj = pickle.load(fh)
    steps = len(obj.named_steps) if isinstance(obj, Pipeline) else -1
    if steps > best_steps:
        best_model, best_steps = obj, steps

if best_model is None or not isinstance(best_model, Pipeline):
    raise TypeError("No saved sklearn Pipeline found among declared model inputs")

target_col = targets[0] if len(targets) == 1 else None
if target_col is None:
    raise ValueError("Permutation importance requires a single declared target column")

y_val = val[target_col]
X_val = val.loc[:, features]

n_classes = task.get("n_classes") or len(np.unique(y_val))
is_multiclass = task.get("type") == "multiclass" or n_classes > 2
if is_multiclass:
    labels = np.unique(y_val)
    scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovr",
        average="macro",
        labels=labels,
    )
else:
    scorer = make_scorer(roc_auc_score, response_method="predict_proba")

result = permutation_importance(
    best_model,
    X_val,
    y_val,
    scoring=scorer,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

imp = pd.DataFrame(
    {
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

for col in ("importance_mean", "importance_sd"):
    imp[col] = imp[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows (multiclass macro OVR ROC AUC, 20 repeats).",
)
top_feature = str(imp.iloc[0]["feature"])
mlca.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"features={len(features)} validation_rows={len(X_val)}")
print(f"top_feature={top_feature} importance_mean={imp.iloc[0]['importance_mean']}")
