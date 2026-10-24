import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc as auc_score,
    balanced_accuracy_score, brier_score_loss,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {list(proto)}")

task = proto["task"]
primary = task["primary_metric"]
targets = list(proto["targets"])
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = targets[0]

ex = proto["excluded_columns"]
ex_cols = set(ex.get("columns", ex.get("examples", []))) if isinstance(ex, dict) else set(ex)
group = proto.get("group_column")

def keylike(s):
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return s.nunique(dropna=True) == len(s) and not s.isna().any()
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=True) == len(s) and any(t in s.name.lower() for t in ("id", "code", "index", "key"))
    return False

base_features = [
    c for c in train.columns
    if c not in set(targets) and c not in ex_cols and c != group and not keylike(train[c])
]

def load(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load("05_preprocess_fit_preprocessing_R1GAM795/preprocessor.pkl")
sel = load("06_features_select_features_FJMHG7YC/selector.pkl")
models = [
    ("07_model_train_logistic_NWPYRQ9N", load("07_model_train_logistic_NWPYRQ9N/model.pkl")),
    ("08_model_train_alternative_FTNXTXVW", load("08_model_train_alternative_FTNXTXVW/alternative_model.pkl")),
]

def fni(obj):
    return list(getattr(obj, "feature_names_in_", []))

def accepts_raw(obj, raw_set):
    names = fni(obj)
    if names:
        return set(names) == raw_set
    if isinstance(obj, Pipeline):
        return accepts_raw(obj.steps[0][1], raw_set)
    return False

def final_est(obj):
    while isinstance(obj, Pipeline):
        obj = obj.steps[-1][1]
    return obj

def input_names(obj):
    names = fni(obj)
    if names:
        return names
    if isinstance(obj, Pipeline):
        return input_names(obj.steps[0][1])
    return None

def make_candidate(model, pre, sel, raw_set):
    if accepts_raw(model, raw_set):
        return model
    if sel is not None and accepts_raw(sel, raw_set):
        front = sel
    elif pre is not None and sel is not None:
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
    elif pre is not None:
        front = pre
    elif sel is not None:
        front = sel
    else:
        return model
    return Pipeline([("frontend", front), ("model", model)])

pcols = fni(pre)
if pcols:
    missing = [c for c in pcols if c not in train.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing columns: {missing}")
    if set(pcols) & set(targets):
        raise ValueError("Target column in preprocessor inputs")
    feature_cols = pcols
else:
    feature_cols = base_features

if set(feature_cols) & set(targets):
    raise ValueError("Target column in feature list")

raw_set = set(feature_cols)
y_val = val[target]
rows = []
primary_values = {}
auc_values = {}

def rs(x):
    return float(f"{float(x):.4g}") if pd.notna(x) else np.nan

for name, model in models:
    cand = make_candidate(model, pre, sel, raw_set)
    names = input_names(cand) or feature_cols
    X = val[list(names)]
    classes = list(final_est(cand).classes_)
    pos = task.get("positive_label")
    if pos is None:
        pos = classes[1]
    elif pos not in classes:
        for c in classes:
            if str(c) == str(pos):
                pos = c
                break
    if pos not in classes:
        raise ValueError(f"Positive label {pos!r} not in classes {classes}")

    p = cand.predict_proba(X)[:, classes.index(pos)]
    yb = (y_val == pos).astype(int).to_numpy()
    if len(np.unique(yb)) < 2:
        raise ValueError("Validation partition lacks both classes")

    roc_auc = roc_auc_score(yb, p)
    prec, rec, _ = precision_recall_curve(yb, p)
    pr_auc = auc_score(rec, prec)
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr
    valid = np.isfinite(thr)
    threshold = float(thr[valid][np.argmax(j[valid])]) if valid.any() else 0.5
    pred = (p >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(yb, pred)
    brier = brier_score_loss(yb, p)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
    }
    if primary not in metrics:
        raise KeyError(f"Primary metric {primary!r} not computed")
    primary_values[name] = metrics[primary]
    auc_values[name] = roc_auc
    mlca_runtime.record_metric(primary, float(metrics[primary]), "val")

    for metric, value in metrics.items():
        rows.append({
            "model": name,
            "metric": metric,
            "value": rs(value),
            "threshold": rs(threshold),
        })

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")

higher_better = primary not in ("brier_score", "log_loss", "mae", "rmse")
best_name = max(primary_values, key=primary_values.get) if higher_better else min(primary_values, key=primary_values.get)
best_val_roc_auc = rs(auc_values[best_name])

mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val_roc_auc, source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("models", len(models))
print("best", best_name, primary, best_val_roc_auc)
