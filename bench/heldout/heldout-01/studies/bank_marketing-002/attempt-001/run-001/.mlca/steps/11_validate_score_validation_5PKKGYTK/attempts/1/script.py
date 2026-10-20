import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
raw_cols = [c for c in train.columns if c not in drop]
for c in list(raw_cols):
    s = train[c]
    if s.dtype.kind in "OU" and s.nunique(dropna=False) == len(s):
        raw_cols.remove(c)

with open(inputs["06_preprocess_fit_preprocessing_STWKFGVD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_N6F5BDYX/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

candidates = [
    ("logistic", inputs["08_model_train_logistic_Y7873DWF/model.pkl"]),
    ("alternative", inputs["09_model_train_alternative_F9H6TZDH/alternative_model.pkl"]),
]

def names_of(obj):
    return [str(c) for c in getattr(obj, "feature_names_in_", [])]

def predict_proba_for(model, Xraw):
    mnames = names_of(model)
    if mnames and set(mnames).issubset(set(Xraw.columns)):
        return model.predict_proba(Xraw[mnames])
    snames = names_of(sel)
    if snames and set(snames).issubset(set(raw_cols)):
        Xs = sel.transform(Xraw[snames])
    else:
        pnames = names_of(pre) or raw_cols
        Xp = pre.transform(Xraw[pnames])
        Xs = sel.transform(Xp) if sel is not None else Xp
    mnames = names_of(model)
    if mnames and hasattr(Xs, "columns"):
        Xs = Xs[mnames]
    return model.predict_proba(Xs)

def classes_of(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        return list(list(model.named_steps.values())[-1].classes_)
    raise AttributeError("Model has no classes_")

y_val = val[targets[0]]
rows = []
val_auc = {}
for name, path in candidates:
    with open(path, "rb") as f:
        model = pickle.load(f)
    classes = classes_of(model)
    pos_label = task.get("positive_label", classes[1])
    pos_idx = classes.index(pos_label)
    proba = predict_proba_for(model, val)
    proba_pos = proba if proba.ndim == 1 else proba[:, pos_idx]
    y_bin = (y_val == pos_label).astype(int)
    auc = roc_auc_score(y_bin, proba_pos)
    ap = average_precision_score(y_bin, proba_pos)
    fpr, tpr, thr = roc_curve(y_bin, proba_pos)
    youden = float(thr[int(np.argmax(tpr - fpr))])
    y_pred = (proba_pos >= youden).astype(int)
    bal = balanced_accuracy_score(y_bin, y_pred)
    brier = brier_score_loss(y_bin, proba_pos)
    val_auc[name] = auc
    rows.extend([
        [name, primary_metric, auc, youden],
        [name, "pr_auc", ap, youden],
        [name, "balanced_accuracy", bal, youden],
        [name, "brier_score", brier, youden],
    ])

metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
metrics["value"] = metrics["value"].astype(float).round(4)
metrics["threshold"] = metrics["threshold"].astype(float).round(4)
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics")

prim = metrics[metrics["metric"] == primary_metric]
lower_better = primary_metric in {"brier_score", "log_loss", "mse", "mae", "rmse", "mape"}
best_row = prim.sort_values("value", ascending=lower_better).iloc[0]
best_name = best_row["model"]
best_auc = float(prim.loc[prim["model"] == best_name, "value"].iloc[0])

mlca_runtime.record_metric(primary_metric, best_auc, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"Candidates: {len(candidates)}")
print(f"Best model: {best_name}")
print(f"Best {primary_metric}: {best_auc}")
