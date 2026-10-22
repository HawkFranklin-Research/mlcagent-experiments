import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def required(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d)}")
    return d[key]


targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
task = required(protocol, "task")
primary = required(task, "primary_metric")
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

protected = set(targets) | set(excluded)
if group:
    protected.add(group)

feature_cols = []
for c in train.columns:
    if c in protected:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_K8ATBX97/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

candidates = {
    "logistic": pickle.load(open(inputs["07_model_train_logistic_CT3F59V6/model.pkl"], "rb")),
    "alternative": pickle.load(open(inputs["latest_model"], "rb")),
}


def make_predictor(obj):
    if isinstance(obj, Pipeline) or hasattr(obj, "feature_names_in_"):
        return obj
    steps = []
    if pre is not None:
        steps.append(pre)
    if sel is not None:
        if isinstance(sel, Pipeline) and any(s is pre for s in sel.named_steps.values()):
            steps = [sel]
        else:
            steps.append(sel)
    steps.append(obj)
    return Pipeline([(f"step{i}", s) for i, s in enumerate(steps)])


def get_feature_names(obj):
    if isinstance(obj, Pipeline):
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
        first = obj.named_steps[list(obj.named_steps)[0]]
        return list(getattr(first, "feature_names_in_", []))
    return list(getattr(obj, "feature_names_in_", []))


def input_columns(obj):
    cols = get_feature_names(obj)
    cols = [c for c in cols if c in feature_cols]
    return cols if cols else feature_cols


def classes_of(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline):
        last = obj.named_steps[list(obj.named_steps)[-1]]
        return np.asarray(getattr(last, "classes_"))
    return np.asarray(getattr(obj, "classes_"))


def eval_model(obj):
    predictor = make_predictor(obj)
    cols = input_columns(predictor)
    proba = predictor.predict_proba(val.loc[:, cols])
    classes = classes_of(predictor)
    declared_pos = protocol.get("positive_label", task.get("positive_label", None))
    pos_label = declared_pos if declared_pos in classes else classes[-1]
    pos_idx = int(np.where(classes == pos_label)[0][0])
    p = proba[:, pos_idx]
    y = val[target].to_numpy()
    yb = (y == pos_label).astype(int)
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr
    finite = np.isfinite(thr)
    idx = np.where(finite)[0][np.argmax(j[finite])] if finite.any() else int(np.argmax(j))
    threshold = float(thr[idx])
    y_pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(yb, p),
        "pr_auc": average_precision_score(yb, p),
        "balanced_accuracy": balanced_accuracy_score(yb, y_pred),
        "brier_score": brier_score_loss(yb, p),
        "threshold": threshold,
        "pos_label": pos_label,
    }


def sig4(x):
    return float(f"{float(x):.4g}")


metrics_by_model = {name: eval_model(obj) for name, obj in candidates.items()}
rows = []
for name, m in metrics_by_model.items():
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]:
        rows.append({
            "model": name,
            "metric": metric,
            "value": sig4(m[metric]),
            "threshold": sig4(m["threshold"]),
        })
metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model.")

lower_better = primary in {"brier_score", "brier", "log_loss", "rmse", "mae"}
best_name = min(metrics_by_model, key=lambda n: metrics_by_model[n][primary]) if lower_better else max(
    metrics_by_model, key=lambda n: metrics_by_model[n][primary]
)
for name, m in metrics_by_model.items():
    mlca_runtime.record_metric(primary, sig4(m[primary]), "val")
mlca_runtime.record_metric(primary, sig4(metrics_by_model[best_name][primary]), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("best_val_roc_auc", sig4(metrics_by_model[best_name]["roc_auc"]), source="validation_metrics.csv")

print("val rows", len(val), "features", len(feature_cols))
print("models", list(candidates))
print("best", best_name, primary, sig4(metrics_by_model[best_name][primary]))
