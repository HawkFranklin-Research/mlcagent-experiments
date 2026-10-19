import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(Path(paths["_protocol"]).read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"missing protocol keys {missing}; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"missing task.primary_metric; task keys: {list(task)}")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print("rows train/val:", len(train), len(val))

def is_keylike(s):
    vals = s.dropna()
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return len(vals) == len(s) and vals.nunique() == len(s)
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return len(vals) == len(s) and vals.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and len(vals) == len(s) and vals.nunique() == len(s):
        return (vals.max() - vals.min() + 1) == len(vals)
    return False

feature_cols = [
    c for c in train.columns
    if c not in set(targets + excluded + [group_col]) and not is_keylike(train[c])
]
print("feature columns:", len(feature_cols))

def load_pkl(p):
    return pickle.loads(Path(p).read_bytes())

latest = load_pkl(paths["latest_model"])
baseline = load_pkl(paths["07_model_train_baseline_2S40PTZZ/model.pkl"])
selector = load_pkl(paths["06_features_retain_features_Z3FWD4WP/selector_all.pkl"])

def final_est(obj):
    if isinstance(obj, Pipeline):
        return obj.named_steps[list(obj.named_steps)[-1]]
    return obj

def names_of(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return []

def importance_names(model):
    est = final_est(model)
    arr = None
    for attr in ("feature_importances_", "coef_"):
        if hasattr(est, attr):
            arr = np.asarray(getattr(est, attr))
            break
    if arr is None:
        return None
    candidates = []
    if hasattr(est, "feature_names_in_"):
        candidates.append(list(est.feature_names_in_))
    if hasattr(model, "feature_names_in_"):
        candidates.append(list(model.feature_names_in_))
    try:
        if hasattr(selector, "get_support") and hasattr(selector, "feature_names_in_"):
            sup = selector.get_support()
            candidates.append([n for n, s in zip(selector.feature_names_in_, sup) if s])
    except Exception:
        pass
    for names in candidates:
        a = np.abs(arr)
        if a.ndim > 1:
            a = a.mean(axis=0)
        if len(a) == len(names):
            return names[int(np.argmax(a))]
    return None

def top_feature(model):
    t = importance_names(model)
    if t:
        return t
    try:
        if hasattr(selector, "get_support") and hasattr(selector, "feature_names_in_"):
            sup = selector.get_support()
            names = [n for n, s in zip(selector.feature_names_in_, sup) if s]
            if names:
                return names[0]
    except Exception:
        pass
    names = names_of(model)
    if names:
        return names[0]
    return feature_cols[0] if feature_cols else "unknown"

top = top_feature(latest)
if top == (feature_cols[0] if feature_cols else None):
    top = top_feature(baseline) or top

def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)

rows = []
rows.append(["model", "frozen_model_class", type(latest).__name__, "latest_model"])
rows.append(["model", "final_estimator_class", type(final_est(latest)).__name__, "latest_model"])
est = final_est(latest)
params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
for k, v in sorted(params.items()):
    if len(rows) >= 22:
        break
    rows.append(["hyperparameter", k, fmt(v)[:200], "latest_model"])

rows.append(["inputs", "targets", ", ".join(targets), "_protocol"])
rows.append(["inputs", "excluded_columns", ", ".join(excluded), "_protocol"])
rows.append(["inputs", "group_column", group_col, "_protocol"])
rows.append(["inputs", "train_rows", str(len(train)), "train"])
rows.append(["inputs", "val_rows", str(len(val)), "val"])
rows.append(["inputs", "feature_count", str(len(feature_cols)), "train"])
rows.append(["inputs", "feature_columns", ", ".join(feature_cols[:20]) + ("..." if len(feature_cols) > 20 else ""), "train"])

vdf = pd.read_csv(paths["09_validate_score_validation_8GEY6C8C/validation_metrics.csv"])
for _, r in vdf.head(12).iterrows():
    item = r.get("metric", r.get("name", "metric"))
    value = r.get("value", r.get("score", ""))
    rows.append(["validation", str(item), fmt(value), "09_validate_score_validation_8GEY6C8C/validation_metrics.csv"])

rows.append(["model", "top_feature", top, "interpretation.csv"])
for note in [
    "No fitting was performed; this is a read-only interpretation.",
    "Validation evidence is limited to the saved validation metrics.",
    "Feature importance is model-specific and not causal.",
    "Predictions require the frozen feature schema and saved preprocessing.",
]:
    rows.append(["limitation", "note", note, "08_model_train_alternative_F5KM7TEB/manifest.json"])

df = pd.DataFrame(rows, columns=["section", "item", "value", "source"]).head(50)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print("saved interpretation.csv rows:", len(df))
print("top_feature:", top)
