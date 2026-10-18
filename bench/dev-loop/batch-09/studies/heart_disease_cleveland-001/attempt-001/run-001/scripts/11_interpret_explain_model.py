import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available top-level keys: {list(d)}")
    return d[key]


targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

blocked = set(targets) | set(excluded) | ({group} if group else set())


def is_key_like(s):
    if s.nunique(dropna=True) != len(s):
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )


raw_features = [
    c for c in train.columns
    if c not in blocked and not is_key_like(train[c])
]

candidates = [
    "latest_model",
    "07_model_train_logistic_J5YKBG0Y/model.pkl",
    "08_model_train_alternative_P5WQW45K/alternative_model.pkl",
    "06_features_select_features_CZZQV6ME/selector.pkl",
    "05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl",
]
best, best_score, best_key = None, (-1, -1), None
for key in candidates:
    try:
        obj = pickle.loads(Path(inputs[key]).read_bytes())
    except Exception:
        continue
    if not isinstance(obj, Pipeline) or not hasattr(obj, "predict_proba"):
        continue
    names = list(getattr(obj, "feature_names_in_", []))
    if names and not set(names).issubset(set(raw_features)):
        continue
    score = (len(obj.named_steps), len(names))
    if score > best_score:
        best, best_score, best_key = obj, score, key

if best is None:
    raise RuntimeError("No complete saved pipeline found")

pipe = best
pipe_cols = list(getattr(pipe, "feature_names_in_", []))
features = pipe_cols if pipe_cols else raw_features
bad = [c for c in features if c in blocked]
if bad:
    raise ValueError(f"Pipeline input contains excluded/target columns: {bad}")
missing = [c for c in features if c not in val.columns]
if missing:
    raise KeyError(f"Validation missing pipeline input columns: {missing}")

X = val[features].copy()
y = val[targets[0]].copy()
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]

if task.get("type") == "binary" or len(pd.unique(y)) <= 2:
    scoring = "roc_auc"
else:
    scoring = "roc_auc_ovr"

result = permutation_importance(
    pipe, X, y, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)
means = np.asarray(result.importances_mean)
sds = np.asarray(result.importances_std)
order = np.argsort(means)[::-1][:15]


def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan


out = pd.DataFrame({
    "feature": [features[i] for i in order],
    "importance_mean": [sig4(means[i]) for i in order],
    "importance_sd": [sig4(sds[i]) for i in order],
})
mlca_runtime.save_table(
    out, "interpretation.csv", caption="Permutation importance on validation"
)

if len(out):
    mlca_runtime.record_fact_candidate(
        "top_feature", out.iloc[0]["feature"], source="interpretation.csv"
    )

print(f"features={len(features)} val_rows={len(X)}")
print(f"pipeline={best_key} top_feature={out.iloc[0]['feature'] if len(out) else 'none'}")
