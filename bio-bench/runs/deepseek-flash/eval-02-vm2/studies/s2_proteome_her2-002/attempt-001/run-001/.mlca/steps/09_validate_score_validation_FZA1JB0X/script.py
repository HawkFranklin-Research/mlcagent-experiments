import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
targets = list(protocol["targets"])
target = targets[0]
task = protocol["task"]
primary = task["primary_metric"]
n_classes = int(task.get("n_classes", train[target].nunique()))
positive = task.get("positive_label") or task.get("positive_class")

def load_pkl(k):
    with open(inputs[k], "rb") as f:
        return pickle.load(f)

preproc = load_pkl("05_preprocess_fit_preprocessing_F5B5FT5N/preprocessor.pkl")
selector = load_pkl("06_features_select_features_ZZ9TRA69/selector.pkl")
models = {
    "logistic": load_pkl("07_model_train_logistic_MCZZ4993/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_B06XN95P/alternative_model.pkl"),
}

feature_cols = None
for obj in (preproc, selector, *models.values()):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        feature_cols = [c for c in list(cols) if c not in set(targets)]
        break
if feature_cols is None:
    excluded = set(protocol.get("excluded_columns", []))
    group = protocol.get("group_column")
    if group:
        excluded.add(group)
    feature_cols = []
    for c in train.columns:
        if c in set(targets) or c in excluded or c not in val.columns:
            continue
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
            continue
        feature_cols.append(c)
feature_cols = [c for c in feature_cols if c in train.columns and c in val.columns]
Xval = val[feature_cols]

def build_frontend(model):
    if isinstance(model, Pipeline):
        return model
    if hasattr(model, "feature_names_in_") and set(model.feature_names_in_) == set(feature_cols):
        return model
    steps = []
    sel_names = getattr(selector, "feature_names_in_", None)
    if sel_names is not None and set(sel_names) <= set(feature_cols):
        steps.append(("selector", selector))
    else:
        steps.append(("preprocess", preproc))
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)

rows = []
primary_vals = {}
auc_vals = {}
for name, model in models.items():
    est = build_frontend(model)
    proba = est.predict_proba(Xval)
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = np.unique(train[target])
    classes = np.asarray(classes)
    if n_classes == 2:
        pos = classes[1] if positive is None else positive
        idx = int(np.where(classes == pos)[0][0])
        yb = (val[target].to_numpy() == pos).astype(int)
        p = proba[:, idx]
        fpr, tpr, thr = roc_curve(yb, p)
        ok = np.isfinite(thr)
        thr = thr[ok]
        j = (tpr - fpr)[ok]
        ythr = float(thr[int(np.argmax(j))]) if len(thr) else 0.5
        predb = (p >= ythr).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(yb, p),
            "pr_auc": average_precision_score(yb, p),
            "balanced_accuracy": balanced_accuracy_score(yb, predb),
            "brier_score": brier_score_loss(yb, p),
        }
        for metric, value in metrics.items():
            rows.append({"model": name, "metric": metric, "value": round(float(value), 4),
                         "threshold": round(ythr, 4) if metric == "balanced_accuracy" else np.nan})
    else:
        present = [c for c in classes if c in set(val[target])]
        aucs = []
        for c in present:
            j = int(np.where(classes == c)[0][0])
            yc = (val[target].to_numpy() == c).astype(int)
            aucs.append(roc_auc_score(yc, proba[:, j]))
        auc = float(np.mean(aucs)) if aucs else np.nan
        pred = classes[np.argmax(proba, axis=1)]
        f1 = f1_score(val[target], pred, average="macro", labels=present)
        metrics = {"roc_auc_ovr": auc, "f1_macro": f1}
        for metric, value in metrics.items():
            rows.append({"model": name, "metric": metric, "value": round(float(value), 4), "threshold": np.nan})
    primary_vals[name] = float(metrics.get(primary, np.nan))
    auc_vals[name] = float(metrics.get("roc_auc", metrics.get("roc_auc_ovr", np.nan)))

lower = any(k in primary.lower() for k in ("brier", "loss", "error", "mse"))
valid_names = [n for n, v in primary_vals.items() if not pd.isna(v)]
best_name = max(valid_names, key=lambda n: primary_vals[n]) if not lower else min(valid_names, key=lambda n: primary_vals[n])
best_primary = primary_vals[best_name]
best_auc = auc_vals[best_name]

out = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_metric(primary, best_primary, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best={best_name} {primary}={best_primary:.4f}")
