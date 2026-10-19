import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available: {list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available: {list(task.keys())}")
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError("Expected exactly one target")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets) | set(excluded)
if group is not None and group != "":
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    lc = str(c).lower()
    if lc in ("id", "index", "row_id") or lc.endswith("_id"):
        continue
    if (pd.api.types.is_string_dtype(s) or s.dtype == object) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X_val = val[feature_cols].copy()
y_val = val[target]

def load_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            import joblib
            f.seek(0)
            return joblib.load(f)

prep = load_pkl(inp["05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_HGTRMBKM/selector.pkl"])

model_keys = [
    ("logistic", "07_model_train_logistic_X3NX87MM/model.pkl"),
    ("latest_model", "latest_model"),
    ("alternative", "08_model_train_alternative_5Z1FXR5Y/alternative_model.pkl"),
]
seen = {}
for label, key in model_keys:
    p = inp[key].resolve()
    if p not in seen:
        seen[p] = label

def get_proba_with(model_obj, X):
    candidates = [("direct", model_obj)]
    if sel is not None:
        candidates.append(("sel+model", Pipeline([("selector", sel), ("model", model_obj)])))
    if prep is not None and sel is not None:
        candidates.append(("prep+sel+model", Pipeline([("preprocessor", prep), ("selector", sel), ("model", model_obj)])))
    if prep is not None:
        candidates.append(("prep+model", Pipeline([("preprocessor", prep), ("model", model_obj)])))
    last_err = None
    for name, cand in candidates:
        try:
            proba = cand.predict_proba(X)
            if proba.shape[0] == len(X) and proba.ndim == 2:
                return cand, proba
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not predict with any pipeline composition: {last_err}")

def get_classes(cand):
    if hasattr(cand, "classes_"):
        return np.asarray(cand.classes_)
    if isinstance(cand, Pipeline):
        final = cand.steps[-1][1]
        if hasattr(final, "classes_"):
            return np.asarray(final.classes_)
    raise AttributeError("No classes_ available")

def round_sig(x):
    if pd.isna(x):
        return np.nan
    return float(f"{x:.4g}")

rows = []
best_label = None
best_value = None
lower_is_better = primary_metric in {"brier_score", "log_loss", "mse", "rmse", "mae"}

for p, label in seen.items():
    model_obj = load_pkl(p)
    cand, proba = get_proba_with(model_obj, X_val)
    classes = get_classes(cand)
    if len(classes) != 2:
        raise ValueError(f"Expected binary classes, got {classes}")
    pos_label = protocol.get("positive_label", task.get("positive_label", None))
    if pos_label is None:
        pos_label = classes[1]
    if pos_label not in classes:
        raise ValueError(f"Positive label {pos_label} not in classes {classes}")
    pos_idx = int(np.where(classes == pos_label)[0][0])
    proba_pos = proba[:, pos_idx]
    y_true = (y_val == pos_label).astype(int).values

    auc = roc_auc_score(y_true, proba_pos)
    ap = average_precision_score(y_true, proba_pos)
    brier = brier_score_loss(y_true, proba_pos)
    fpr, tpr, thr = roc_curve(y_true, proba_pos)
    youden_thr = float(thr[int(np.argmax(tpr - fpr))])
    y_pred = (proba_pos >= youden_thr).astype(int)
    bal = balanced_accuracy_score(y_true, y_pred)

    values = {"roc_auc": auc, "pr_auc": ap, "balanced_accuracy": bal, "brier_score": brier}
    if primary_metric not in values:
        raise ValueError(f"Unsupported primary_metric {primary_metric}")
    pv = float(values[primary_metric])
    mlca_runtime.record_metric(primary_metric, pv, split="val")

    rows.append({"model": label, "metric": "roc_auc", "value": auc, "threshold": np.nan})
    rows.append({"model": label, "metric": "pr_auc", "value": ap, "threshold": np.nan})
    rows.append({"model": label, "metric": "balanced_accuracy", "value": bal, "threshold": youden_thr})
    rows.append({"model": label, "metric": "brier_score", "value": brier, "threshold": np.nan})

    if best_value is None or (pv < best_value if lower_is_better else pv > best_value):
        best_value = pv
        best_label = label

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(round_sig)
metrics_df["threshold"] = metrics_df["threshold"].map(round_sig)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics for candidate models")

best_auc = float(metrics_df.loc[(metrics_df["model"] == best_label) & (metrics_df["metric"] == "roc_auc"), "value"].iloc[0])
mlca_runtime.record_fact_candidate("best_model", best_label, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train_rows={len(train)} val_rows={len(val)} features={len(feature_cols)}")
print(f"best_model={best_label} {primary_metric}={best_value:.4g} val_roc_auc={best_auc:.4g}")
