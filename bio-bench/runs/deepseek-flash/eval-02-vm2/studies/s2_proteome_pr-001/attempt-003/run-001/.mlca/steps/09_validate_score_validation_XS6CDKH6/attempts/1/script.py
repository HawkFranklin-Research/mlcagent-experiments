import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def load_pickle(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ("model", "estimator", "pipeline", "preprocessor", "selector"):
            if k in obj:
                return obj[k]
    return obj

def req(d, key, ctx):
    if key not in d:
        raise KeyError(f"Missing {key!r} in {ctx}; available keys: {list(d)}")
    return d[key]

proto = json.loads(Path(inputs["_protocol"]).read_text())
task = req(proto, "task", "_protocol")
primary = req(task, "primary_metric", "_protocol.task")
targets = req(proto, "targets", "_protocol")
if isinstance(targets, str):
    try:
        targets = json.loads(targets)
    except Exception:
        targets = [targets]
excluded = req(proto, "excluded_columns", "_protocol")
if not isinstance(excluded, (list, tuple, set)):
    raise TypeError(f"excluded_columns must be a list; got {type(excluded)}; top-level keys: {list(proto)}")
group = proto.get("group_column", None)
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

tgt = set(targets)
exc = set(excluded)
base = [c for c in train.columns if c not in tgt and c not in exc and c != group]
feature_cols = []
for c in base:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype) or pd.api.types.is_integer_dtype(s):
        if s.notna().all() and s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

pre = unwrap(load_pickle("05_preprocess_fit_preprocessing_GK4CB9FZ/preprocessor.pkl"))
sel = unwrap(load_pickle("06_features_select_features_5542Z2W2/selector.pkl"))
cands = [
    ("logistic", unwrap(load_pickle("07_model_train_logistic_VYQ2HW60/model.pkl"))),
    ("alternative", unwrap(load_pickle("08_model_train_alternative_J88G12HR/alternative_model.pkl"))),
]

def proba_for(obj, frame, name):
    if not hasattr(obj, "predict_proba"):
        raise TypeError(f"{name} has no predict_proba")
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
        if all(c in frame.columns for c in cols):
            return obj.predict_proba(frame[cols])
        if isinstance(obj, Pipeline):
            missing = [c for c in cols if c not in frame.columns]
            raise KeyError(f"{name}: missing input columns {missing[:5]}")
    if isinstance(sel, Pipeline):
        steps = list(sel.named_steps.values())
        front = sel if any(step is pre for step in steps) else Pipeline([("preprocessor", pre), ("selector", sel)])
    else:
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
    pipe = Pipeline([("frontend", front), ("model", obj)])
    return pipe.predict_proba(frame[feature_cols])

def metrics_for(name, obj):
    proba = proba_for(obj, val, name)
    classes = list(obj.classes_)
    if len(classes) != 2:
        raise ValueError(f"{name}: expected binary classes, got {classes}")
    pos = task.get("positive_label", None)
    if pos is None:
        pos = classes[1]
    if pos not in classes:
        raise ValueError(f"{name}: positive label {pos!r} not in {classes}")
    p = proba[:, classes.index(pos)]
    y = val[target_col].to_numpy()
    yb = (y == pos).astype(int)
    auc = roc_auc_score(yb, p)
    ap = average_precision_score(yb, p)
    brier = brier_score_loss(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr
    finite = np.isfinite(thr)
    if finite.any():
        ids = np.flatnonzero(finite)
        ti = ids[int(np.argmax(j[finite]))]
    else:
        ti = int(np.argmax(j))
    threshold = float(thr[ti])
    ba = balanced_accuracy_score(yb, (p >= threshold).astype(int))
    return {"roc_auc": auc, "pr_auc": ap, "balanced_accuracy": ba, "brier_score": brier}, threshold

rows = []
per_model = {}
for name, obj in cands:
    m, thr = metrics_for(name, obj)
    per_model[name] = m
    for metric, value in m.items():
        rows.append((name, metric, value, thr if metric == "balanced_accuracy" else np.nan))

alias = {"roc_auc": "roc_auc", "pr_auc": "pr_auc", "average_precision": "pr_auc",
         "balanced_accuracy": "balanced_accuracy", "brier_score": "brier_score"}
pkey = alias.get(primary, primary)
if pkey not in per_model[next(iter(per_model))]:
    raise KeyError(f"Unsupported primary metric {primary!r}")
lower_better = {"brier_score", "log_loss", "rmse", "mae"}
best_name = (min if primary in lower_better else max)(per_model, key=lambda n: per_model[n][pkey])
best_auc = per_model[best_name]["roc_auc"]

def sig(x):
    return float(f"{x:.4g}") if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan

rows = [(n, m, sig(v), sig(t)) for n, m, v, t in rows]
metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics for candidate models.")

order = [n for n, _ in cands if n != best_name] + [best_name]
for n in order:
    mlca_runtime.record_metric(primary, float(per_model[n][pkey]), split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best_model={best_name} {primary}={per_model[best_name][pkey]:.4g} val_roc_auc={best_auc:.4g}")
