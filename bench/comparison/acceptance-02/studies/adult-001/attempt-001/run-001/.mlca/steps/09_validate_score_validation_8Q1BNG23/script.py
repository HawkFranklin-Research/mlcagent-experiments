import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
_protocol = json.loads(inp["_protocol"].read_text())

try:
    targets = _protocol["targets"]
    excluded = _protocol["excluded_columns"]
    task = _protocol["task"]
    metric_name = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(_protocol)}")

group = _protocol.get("group_column")
target = targets[0]
drop = set(targets) | set(excluded) | ({group} if group else set())

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    key_like = (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=True) == len(s)
    if not key_like:
        feature_cols.append(c)

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

paths = [
    ("logistic", inp["07_model_train_logistic_X4STNM7K/model.pkl"]),
    ("alternative", inp["08_model_train_alternative_KMW9HJFX/alternative_model.pkl"]),
]
y = val[target]
rows, res = [], {}

for name, path in paths:
    model = load_pkl(path)
    expected = getattr(model, "feature_names_in_", None)
    cols = list(expected) if expected is not None else feature_cols
    if set(cols) & drop:
        raise ValueError("model expects target or excluded column")
    X = val.loc[:, cols]

    classes = getattr(model, "classes_", None)
    if classes is None:
        last = model.steps[-1][1] if hasattr(model, "steps") else model
        classes = last.classes_
    classes = list(classes)

    pos_label = task.get("positive_label", task.get("positive_class"))
    if pos_label is None:
        pos_label = classes[1]
    p = model.predict_proba(X)[:, classes.index(pos_label)]
    ybin = (y == pos_label).astype(int).to_numpy()

    auc = roc_auc_score(ybin, p)
    ap = average_precision_score(ybin, p)
    brier = brier_score_loss(ybin, p)
    fpr, tpr, thr = roc_curve(ybin, p)
    th = float(thr[int(np.argmax(tpr - fpr))])
    bal = balanced_accuracy_score(ybin, (p >= th).astype(int))

    res[name] = {"roc_auc": float(auc), "pr_auc": float(ap),
                 "balanced_accuracy": float(bal), "brier_score": float(brier)}
    for m, v, t in [("roc_auc", auc, np.nan), ("pr_auc", ap, np.nan),
                    ("balanced_accuracy", bal, th), ("brier_score", brier, np.nan)]:
        rows.append({"model": name, "metric": m, "value": float(v), "threshold": t})
    mlca_runtime.record_metric(metric_name, float(res[name][metric_name]), "val")

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation metrics by candidate model")

if metric_name == "brier_score":
    best = min(res, key=lambda n: res[n][metric_name])
else:
    best = max(res, key=lambda n: res[n][metric_name])

mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(res[best]["roc_auc"]),
                                   source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape}")
print(f"primary={metric_name}; best={best}; val_auc={res[best]['roc_auc']:.4g}")
print(f"metrics_rows={len(metrics_df)}")
