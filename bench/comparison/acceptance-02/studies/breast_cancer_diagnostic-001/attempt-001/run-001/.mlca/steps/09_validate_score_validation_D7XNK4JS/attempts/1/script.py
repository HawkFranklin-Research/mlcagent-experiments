import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)
import mlca_runtime

mlca_runtime.set_seed(42)
P = mlca_runtime.inputs()
protocol = json.loads(P["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing key {key!r}; available top-level keys: {list(d.keys())}")
    return d[key]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
primary_metric = req(task, "primary_metric")
group_col = protocol.get("group_column", None)
target = targets[0]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def accepts_raw(obj, raw_cols):
    if obj is None:
        return False
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names) == list(raw_cols)
    if isinstance(obj, Pipeline):
        steps = list(obj.named_steps.values())
        if steps:
            return accepts_raw(steps[0], raw_cols)
    return False

def build_candidate(model_obj, preproc, sel, raw_cols):
    if accepts_raw(model_obj, raw_cols):
        return model_obj
    if accepts_raw(sel, raw_cols):
        front = sel
    elif accepts_raw(preproc, raw_cols):
        front = Pipeline([("preprocess", preproc), ("selector", sel)])
    else:
        front = Pipeline([("preprocess", preproc), ("selector", sel)])
    return Pipeline([("frontend", front), ("model", model_obj)])

train = pd.read_parquet(P["train"])
val = pd.read_parquet(P["val"])

special = set(targets) | set(excluded)
if group_col is not None:
    special.add(group_col)

raw_cols = []
for c in train.columns:
    if c in special:
        continue
    s = train[c]
    dtype = str(s.dtype)
    if dtype == "object" or dtype.startswith("string") or dtype == "category":
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        vals = np.sort(s.dropna().unique())
        if len(vals) > 1 and np.all(np.diff(vals) == 1):
            continue
    raw_cols.append(c)

X_val = val[raw_cols].copy()
y_val = val[target]

preproc = load_pkl(P["05_preprocess_fit_preprocessing_6Y57Q816/preprocessor.pkl"])
sel = load_pkl(P["06_features_select_features_HZT7Z0Q4/selector.pkl"])
candidates = [
    ("logistic", P["07_model_train_logistic_94MG7111/model.pkl"]),
    ("alternative", P["08_model_train_alternative_227Y318V/alternative_model.pkl"]),
]

rows = []
model_primary = {}
model_roc_auc = {}

for name, path in candidates:
    model_obj = load_pkl(path)
    cand = build_candidate(model_obj, preproc, sel, raw_cols)
    proba = cand.predict_proba(X_val)
    classes = np.asarray(getattr(cand, "classes_", cand.named_steps["model"].classes_))
    if task.get("type", "binary") == "binary":
        pos_label = task.get("positive_label", None)
        if pos_label is None:
            pos_label = classes[1]
        pos_idx = int(np.where(classes == pos_label)[0][0])
        p = proba[:, pos_idx]
        yb = (y_val == pos_label).astype(int)
        if yb.nunique() < 2:
            roc = pr = bal = brier = thr = np.nan
        else:
            roc = roc_auc_score(yb, p)
            pr = average_precision_score(yb, p)
            brier = brier_score_loss(yb, p)
            fpr, tpr, ths = roc_curve(yb, p)
            bi = int(np.argmax(tpr - fpr))
            thr = float(ths[bi])
            if not np.isfinite(thr):
                thr = 1.0
            bal = balanced_accuracy_score(yb, (p >= thr).astype(int))
        metric_vals = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier_score": brier}
        if primary_metric not in metric_vals:
            raise KeyError(f"Unsupported primary metric {primary_metric!r}")
        model_primary[name] = float(metric_vals[primary_metric])
        model_roc_auc[name] = float(roc)
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]:
            rows.append({
                "model": name,
                "metric": metric,
                "value": metric_vals[metric],
                "threshold": thr if metric == "balanced_accuracy" else np.nan,
            })
    else:
        present = [c for c in classes if np.any(y_val == c)]
        if len(present) == len(classes):
            roc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            vals = []
            for c in present:
                j = int(np.where(classes == c)[0][0])
                vals.append(roc_auc_score((y_val == c).astype(int), proba[:, j]))
            roc = float(np.mean(vals))
        pred = cand.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        metric_vals = {"roc_auc": roc, "f1_macro": f1}
        if primary_metric not in metric_vals:
            raise KeyError(f"Unsupported primary metric {primary_metric!r}")
        model_primary[name] = float(metric_vals[primary_metric])
        model_roc_auc[name] = float(roc)
        for metric in ["roc_auc", "f1_macro"]:
            rows.append({"model": name, "metric": metric, "value": metric_vals[metric], "threshold": np.nan})

valid_primary = {k: v for k, v in model_primary.items() if pd.notna(v)}
if not valid_primary:
    raise ValueError("No valid primary metric values on validation")
lower_better = {"brier_score", "log_loss", "mean_squared_error", "rmse", "mae"}
if primary_metric in lower_better:
    best_model = min(valid_primary, key=valid_primary.get)
else:
    best_model = max(valid_primary, key=valid_primary.get)
best_val_primary = model_primary[best_model]
best_val_roc_auc = model_roc_auc[best_model]

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{x:.4g}")

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(sig4)
metrics_df["threshold"] = metrics_df["threshold"].map(sig4)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model.")

mlca_runtime.record_metric(primary_metric, float(best_val_primary), split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_val_roc_auc), source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("features", len(raw_cols), "metric", primary_metric)
print("best", best_model, primary_metric, round(float(best_val_primary), 4))
print("saved validation_metrics.csv")
