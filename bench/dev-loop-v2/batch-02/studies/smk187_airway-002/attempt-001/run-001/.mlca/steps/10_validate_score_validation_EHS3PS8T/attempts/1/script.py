import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def require(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}. Available top-level keys: {list(protocol.keys())}")
    return d[key]

targets = require(protocol, "targets")
excluded = require(protocol, "excluded_columns")
task = require(protocol, "task")
primary = require(task, "primary_metric")
group = protocol["group_column"] if "group_column" in protocol else None
task_type = task.get("type", "binary")
positive_label = task.get("positive_label", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
if len(targets) != 1:
    raise ValueError("This validation script expects exactly one target column.")
target_col = targets[0]
y_train = train[target_col].to_numpy()
y_val = val[target_col].to_numpy()

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype.kind in "OUS" and s.nunique(dropna=False) == len(train):
        continue
    if s.dtype.kind in "iu" and s.nunique(dropna=False) == len(train):
        continue
    feature_cols.append(c)

X_val = val[feature_cols].copy()
raw_set = set(feature_cols)

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_XWW5WTFN/preprocessor.pkl")
sel = load_pkl("07_features_select_features_3JX8FX9S/selector.pkl")
models = {
    "08_model_train_logistic_1JG6Z9GW": load_pkl("08_model_train_logistic_1JG6Z9GW/model.pkl"),
    "09_model_train_alternative_N0C35K8C": load_pkl("09_model_train_alternative_N0C35K8C/alternative_model.pkl"),
}

def input_names(obj):
    if obj is None:
        return None
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        try:
            return list(names)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                try:
                    return list(names)
                except Exception:
                    pass
    return None

def accepts_raw(obj):
    names = input_names(obj)
    if names is None:
        return isinstance(obj, Pipeline) or hasattr(obj, "predict_proba")
    return all(n in raw_set for n in names)

pre_accepts_raw = accepts_raw(pre)
sel_accepts_raw = accepts_raw(sel)
if sel_accepts_raw:
    frontend = sel
elif pre_accepts_raw:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
else:
    frontend = pre

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return list(obj.classes_)
    if isinstance(obj, Pipeline):
        est = obj.steps[-1][1]
        if hasattr(est, "classes_"):
            return list(est.classes_)
    return None

def predict_proba_raw(model_obj, Xraw):
    names = input_names(model_obj)
    if names is not None and all(n in Xraw.columns for n in names):
        try:
            return model_obj.predict_proba(Xraw[names])
        except Exception:
            pass
    if names is None:
        try:
            return model_obj.predict_proba(Xraw[feature_cols])
        except Exception:
            pass
    if not isinstance(model_obj, Pipeline):
        try:
            pipe = Pipeline([("frontend", frontend), ("model", model_obj)])
            return pipe.predict_proba(Xraw[feature_cols])
        except Exception:
            pass
    if isinstance(model_obj, Pipeline) and pre_accepts_raw:
        try:
            pipe = Pipeline([("preprocessor", pre), ("model", model_obj)])
            return pipe.predict_proba(Xraw[feature_cols])
        except Exception:
            pass
    return model_obj.predict_proba(Xraw[feature_cols])

def binary_metrics(y, proba, classes):
    y = np.asarray(y)
    classes = list(classes) if classes is not None else list(np.sort(pd.unique(y)))
    pos = positive_label
    if pos is None:
        pos = classes[1] if len(classes) > 1 else classes[0]
    if pos not in classes and len(classes) > 1:
        pos = classes[1]
    proba = np.asarray(proba)
    if proba.ndim == 1:
        p = proba
    else:
        j = classes.index(pos) if pos in classes else (1 if proba.shape[1] > 1 else 0)
        p = proba[:, j]
    y_bin = (y == pos).astype(int)
    roc = roc_auc_score(y_bin, p)
    pr = average_precision_score(y_bin, p)
    brier = brier_score_loss(y_bin, p)
    fpr, tpr, thr = roc_curve(y_bin, p)
    j_score = tpr - fpr
    finite_idx = np.where(np.isfinite(thr))[0]
    idx = finite_idx[np.argmax(j_score[finite_idx])] if len(finite_idx) else int(np.argmax(j_score))
    threshold = float(thr[idx])
    if not np.isfinite(threshold):
        threshold = 0.5
    pred = (p >= threshold).astype(int)
    bal = balanced_accuracy_score(y_bin, pred)
    metrics = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier_score": brier}
    thresholds = {"balanced_accuracy": threshold}
    return metrics, thresholds

def multiclass_metrics(y, proba, classes):
    y = np.asarray(y)
    proba = np.asarray(proba)
    classes = list(classes) if classes is not None else list(np.sort(pd.unique(y)))
    aucs, missing = [], []
    for j, c in enumerate(classes):
        mask = (y == c)
        if mask.any() and not mask.all():
            aucs.append(roc_auc_score(mask.astype(int), proba[:, j]))
        else:
            missing.append(c)
    macro_auc = float(np.mean(aucs)) if aucs else np.nan
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    macro_f1 = f1_score(y, pred, labels=classes, average="macro", zero_division=0)
    return {"roc_auc": macro_auc, "f1_macro": macro_f1}, missing

rows = []
primary_values = {}
for model_name, model_obj in models.items():
    proba = predict_proba_raw(model_obj, X_val)
    classes = get_classes(model_obj)
    if (proba.ndim > 1 and proba.shape[1] > 2) or (classes is not None and len(classes) > 2):
        metrics, missing = multiclass_metrics(y_val, proba, classes)
        if missing:
            print(f"{model_name}: classes absent from validation AUC: {missing}")
        for metric_name, value in metrics.items():
            rows.append({"model": model_name, "metric": metric_name, "value": value, "threshold": np.nan})
    else:
        metrics, thresholds = binary_metrics(y_val, proba, classes)
        for metric_name, value in metrics.items():
            rows.append({
                "model": model_name,
                "metric": metric_name,
                "value": value,
                "threshold": thresholds.get(metric_name, np.nan),
            })
    if primary not in metrics:
        raise ValueError(f"Primary metric {primary!r} not computed. Available: {list(metrics.keys())}")
    primary_values[model_name] = metrics[primary]

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
def round4(x):
    return np.nan if pd.isna(x) else float(f"{float(x):.4g}")
metrics_df["value"] = metrics_df["value"].map(round4)
metrics_df["threshold"] = metrics_df["threshold"].map(round4)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics")

lower_better = (
    primary.lower() in {"brier_score", "log_loss", "mae", "mse", "rmse"}
    or "loss" in primary.lower()
    or "error" in primary.lower()
)
best_model = min(primary_values, key=primary_values.get) if lower_better else max(primary_values, key=primary_values.get)
best_primary = float(primary_values[best_model])
mlca_runtime.record_metric(primary, best_primary, split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

auc_rows = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["metric"] == "roc_auc")]
if len(auc_rows):
    mlca_runtime.record_fact_candidate(
        "best_val_roc_auc", float(auc_rows.iloc[0]["value"]), source="validation_metrics.csv"
    )

print(f"train={train.shape}, val={val.shape}, features={len(feature_cols)}")
print(f"primary={primary}, best_model={best_model}, best_val={best_primary:.4g}")
