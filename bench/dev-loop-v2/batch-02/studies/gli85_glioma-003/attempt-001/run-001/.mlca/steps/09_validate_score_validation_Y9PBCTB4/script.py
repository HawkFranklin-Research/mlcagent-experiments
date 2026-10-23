import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def pkey(key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
    return protocol[key]

targets = pkey("targets")
excluded = pkey("excluded_columns")
group_column = pkey("group_column")
task = pkey("task")
primary_metric = task["primary_metric"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target_col = targets[0]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def is_key_like(s):
    if not pd.api.types.is_numeric_dtype(s):
        return s.is_unique
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        if s.is_unique and any(tok in name for tok in ("id", "key", "code")):
            return True
        if s.is_unique and len(s) > 1:
            vals = np.sort(s.dropna().to_numpy())
            if len(vals) == len(s) and vals[0] >= 0 and vals[-1] == len(s) - 1:
                return bool(np.array_equal(vals, np.arange(len(s))))
    return False

exclude_set = set(targets) | set(excluded)
if group_column:
    exclude_set.add(group_column)
feature_cols = [c for c in train.columns if c not in exclude_set and not is_key_like(train[c])]

def load_pkl(key):
    with open(paths[key], "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("05_preprocess_fit_preprocessing_W6YRPZJC/preprocessor.pkl")
selector = load_pkl("06_features_select_features_GPEMXWYC/selector.pkl")
models = [
    ("logistic", load_pkl("07_model_train_logistic_9GB0B8SE/model.pkl")),
    ("alternative", load_pkl("08_model_train_alternative_C89NP2WV/alternative_model.pkl")),
]

def frontend():
    if selector is not None:
        names = getattr(selector, "feature_names_in_", None)
        if names is None or all(c in feature_cols for c in names):
            return selector
        if preprocessor is not None:
            return Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    if preprocessor is not None:
        return preprocessor
    return None

def build_predictor(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None and all(c in feature_cols for c in names):
        return obj
    if isinstance(obj, Pipeline):
        if names is None:
            return obj
        if preprocessor is not None and not any(step is preprocessor for _, step in obj.named_steps.items()):
            return Pipeline([("preprocessor", preprocessor), ("model", obj)])
        return obj
    fe = frontend()
    return obj if fe is None else Pipeline([("frontend", fe), ("model", obj)])

def model_X(pred):
    names = getattr(pred, "feature_names_in_", None)
    if names is None:
        names = feature_cols
    missing = [c for c in names if c not in val.columns]
    if missing:
        raise KeyError(f"Validation missing model input columns: {missing[:5]}")
    return val.loc[:, list(names)]

def round4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

y_val = val[target_col]
positive_label = task.get("positive_label", task.get("positive_class", None))
metric_values = {}
rows = []
for name, obj in models:
    pred = build_predictor(obj)
    X = model_X(pred)
    proba = pred.predict_proba(X)
    classes = list(pred.classes_)
    if len(classes) != 2:
        raise ValueError(f"{name}: expected binary classes, got {classes}")
    pos_label = classes[1] if positive_label is None else positive_label
    if pos_label not in classes:
        raise ValueError(f"{name}: positive label {pos_label!r} not in {classes}")
    neg_label = [c for c in classes if c != pos_label][0]
    pos_idx = classes.index(pos_label)
    y_bin = (y_val == pos_label).to_numpy(dtype=int)
    proba_pos = proba[:, pos_idx]

    auc = roc_auc_score(y_bin, proba_pos)
    pr_auc = average_precision_score(y_bin, proba_pos)
    brier = brier_score_loss(y_bin, proba_pos)
    fpr, tpr, thresholds = roc_curve(y_bin, proba_pos)
    thr = float(thresholds[int(np.argmax(tpr - fpr))])
    if not np.isfinite(thr):
        thr = float(np.nextafter(np.max(proba_pos), np.inf))
    y_pred_labels = np.where(proba_pos >= thr, pos_label, neg_label)
    bal = balanced_accuracy_score(y_val.to_numpy(), y_pred_labels)

    metric_values[name] = {
        "roc_auc": float(auc), "pr_auc": float(pr_auc),
        "balanced_accuracy": float(bal), "brier_score": float(brier),
    }
    rows.extend([
        {"model": name, "metric": "roc_auc", "value": round4(auc), "threshold": np.nan},
        {"model": name, "metric": "pr_auc", "value": round4(pr_auc), "threshold": np.nan},
        {"model": name, "metric": "balanced_accuracy", "value": round4(bal), "threshold": round4(thr)},
        {"model": name, "metric": "brier_score", "value": round4(brier), "threshold": np.nan},
    ])

if primary_metric not in {"roc_auc", "pr_auc", "balanced_accuracy", "brier_score"}:
    raise ValueError(f"Unsupported primary_metric {primary_metric!r}")

lower_better = primary_metric in {"brier_score", "log_loss", "mse", "mae", "rmse"}
names = [n for n, _ in models]
if lower_better:
    best_name = min(names, key=lambda n: metric_values[n][primary_metric])
else:
    best_name = max(names, key=lambda n: metric_values[n][primary_metric])
best_value = float(metric_values[best_name][primary_metric])
best_auc = float(metric_values[best_name]["roc_auc"])

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(
    metrics_df, "validation_metrics.csv",
    caption="Validation metrics by model; threshold is validation Youden J for balanced accuracy."
)
mlca_runtime.record_metric(primary_metric, best_value, split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best_model={best_name} {primary_metric}={best_value:.4g}")
