import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
avail = list(protocol)
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {avail}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"] or []
task = protocol["task"]
group = protocol.get("group_column", [])
group = [] if group is None else ([group] if isinstance(group, str) else list(group))
target_col = targets[0]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y_val = val[target_col]

def keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c not in group and not keylike(train[c])
]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if isinstance(model, Pipeline):
        est = model.named_steps[list(model.named_steps)[-1]]
        return np.asarray(est.classes_)
    raise AttributeError("model has no classes_")

def make_full(cand):
    if isinstance(cand, Pipeline) and len(cand.named_steps) > 1:
        return cand
    pre = load_pkl(inp["05_preprocess_fit_preprocessing_VDSDFD6K/preprocessor.pkl"])
    sel = load_pkl(inp["06_features_select_features_C9CJJ4J0/selector.pkl"])
    front = sel if isinstance(sel, Pipeline) and len(sel.named_steps) > 1 else Pipeline([
        ("preprocessor", pre), ("selector", sel)
    ])
    if isinstance(cand, Pipeline):
        return Pipeline([("front", front)] + list(cand.steps))
    return Pipeline([("front", front), ("model", cand)])

def eval_candidate(name, cand):
    full = make_full(cand)
    expected = getattr(full, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        if not set(expected).issubset(set(feature_cols)):
            raise ValueError(f"Unexpected feature columns for {name}: {expected}")
        X = val.loc[:, expected]
    else:
        X = val.loc[:, feature_cols]

    proba = full.predict_proba(X)
    classes = get_classes(full)
    if len(classes) == 2:
        pos_label = task.get("positive_label", None)
        if pos_label is None:
            pos_label = classes[1]
        if pos_label not in classes:
            raise ValueError(f"Declared positive label {pos_label!r} not in model classes")
        pos_idx = int(np.where(classes == pos_label)[0][0])
        neg_label = classes[1 - pos_idx]
        prob_pos = proba[:, pos_idx]
        y_bin = (y_val == pos_label).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, prob_pos)
        j = tpr - fpr
        finite = np.isfinite(thr)
        idx = np.where(finite)[0][np.argmax(j[finite])] if finite.any() else int(np.argmax(j))
        threshold = float(thr[idx])
        pred = np.where(prob_pos >= threshold, pos_label, neg_label)
        metrics = {
            "roc_auc": roc_auc_score(y_bin, prob_pos),
            "pr_auc": average_precision_score(y_bin, prob_pos),
            "balanced_accuracy": balanced_accuracy_score(y_val, pred),
            "brier": brier_score_loss(y_bin, prob_pos),
        }
        rows = [(name, k, v, threshold) for k, v in metrics.items()]
        return metrics, rows, threshold

    roc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    pred = classes[np.argmax(proba, axis=1)]
    metrics = {"roc_auc_ovr": roc, "f1_macro": f1_score(y_val, pred, average="macro")}
    rows = [(name, k, v, np.nan) for k, v in metrics.items()]
    return metrics, rows, np.nan

candidates = {
    "logistic": load_pkl(inp["07_model_train_logistic_NCC1WZ5Q/model.pkl"]),
    "alternative": load_pkl(inp["08_model_train_alternative_BY729WV6/alternative_model.pkl"]),
}
all_rows, metrics_by_name, primary_vals = [], {}, {}
for name, cand in candidates.items():
    metrics, rows, threshold = eval_candidate(name, cand)
    metrics_by_name[name] = metrics
    all_rows.extend(rows)
    if primary_metric in metrics:
        pkey = primary_metric
    elif primary_metric == "roc_auc" and "roc_auc_ovr" in metrics:
        pkey = "roc_auc_ovr"
    else:
        raise KeyError(f"Cannot evaluate declared primary metric {primary_metric!r} for {name}")
    primary_vals[name] = metrics[pkey]
    mlca_runtime.record_metric(primary_metric, float(metrics[pkey]), "val")

direction = {"brier": "low", "log_loss": "low"}.get(primary_metric, "high")
best_name = max(primary_vals, key=primary_vals.get) if direction == "high" else min(primary_vals, key=primary_vals.get)
best_metrics = metrics_by_name[best_name]
best_auc = best_metrics.get("roc_auc", best_metrics.get("roc_auc_ovr"))

def sig(v):
    return np.nan if pd.isna(v) else float(f"{v:.4g}")

out = pd.DataFrame(all_rows, columns=["model", "metric", "value", "threshold"])
out["value"] = out["value"].map(sig)
out["threshold"] = out["threshold"].map(sig)
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", sig(best_auc), source="validation_metrics.csv")

print(f"features={len(feature_cols)} val_rows={len(val)}")
print(f"best_model={best_name} {primary_metric}={primary_vals[best_name]:.4g}")
