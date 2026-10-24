import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available keys: {list(d)}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
primary = req(task, "primary_metric")
if len(targets) != 1:
    raise ValueError("This validation script expects exactly one target")
target = targets[0]
group = protocol.get("group_column")
excl = excluded if not isinstance(excluded, dict) else (excluded.get("columns") or excluded.get("examples") or [])
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
skip = set(targets) | set(excl) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in skip and pd.api.types.is_numeric_dtype(train[c])]

def load_obj(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "estimator"):
            if k in obj:
                return obj[k]
    return obj

def get_classes(m):
    cls = getattr(m, "classes_", None)
    if cls is None and hasattr(m, "named_steps"):
        cls = getattr(list(m.named_steps.values())[-1], "classes_", None)
    if cls is None:
        raise AttributeError("Model exposes no classes_")
    return np.asarray(cls)

def model_columns(m):
    cols = getattr(m, "feature_names_in_", None)
    if cols is None and hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                break
    return list(cols) if cols is not None else feature_cols

models = {
    "logistic": load_obj(inp["07_model_train_logistic_HJ9ARHSV/model.pkl"]),
    "alternative": load_obj(inp["08_model_train_alternative_DNABCDVY/alternative_model.pkl"]),
}
y_val = val[target]
rows = []
auc_vals = {}
primary_vals = {}

for name, model in models.items():
    cols = model_columns(model)
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"{name} missing validation columns: {missing}")
    X = val[cols]
    proba = model.predict_proba(X)
    classes = get_classes(model)

    if len(classes) == 2:
        pos = task.get("positive_label", classes[1])
        if pos not in list(classes):
            raise ValueError(f"Positive label {pos!r} not in model classes {list(classes)}")
        j = list(classes).index(pos)
        yb = (y_val == pos).astype(int).to_numpy()
        p = np.asarray(proba)[:, j]
        fpr, tpr, thr = roc_curve(yb, p)
        J = tpr - fpr
        finite = np.isfinite(thr)
        threshold = float(thr[finite][np.argmax(J[finite])]) if finite.any() else 0.5
        yp = (p >= threshold).astype(int)
        met = {
            "roc_auc": roc_auc_score(yb, p),
            "pr_auc": average_precision_score(yb, p),
            "balanced_accuracy": balanced_accuracy_score(yb, yp),
            "brier": brier_score_loss(yb, p),
        }
        for metric, value in met.items():
            rows.append({"model": name, "metric": metric, "value": value, "threshold": threshold})
        auc_vals[name] = met["roc_auc"]
        primary_vals[name] = met.get(primary)
    else:
        present = [c for c in classes if (y_val == c).any()]
        left_out = [c for c in classes if c not in present]
        aucs = []
        for c in present:
            j = list(classes).index(c)
            yc = (y_val == c).astype(int).to_numpy()
            if len(np.unique(yc)) > 1:
                aucs.append(roc_auc_score(yc, np.asarray(proba)[:, j]))
        auc = float(np.mean(aucs)) if aucs else np.nan
        f1 = f1_score(y_val, model.predict(X), labels=classes, average="macro", zero_division=0)
        rows.append({"model": name, "metric": "roc_auc_ovr", "value": auc, "threshold": np.nan})
        rows.append({"model": name, "metric": "f1_macro", "value": f1, "threshold": np.nan})
        auc_vals[name] = auc
        primary_vals[name] = {"roc_auc_ovr": auc, "f1_macro": f1}.get(primary)
        if left_out:
            print(f"{name} classes absent from validation: {left_out}")

if any(v is None for v in primary_vals.values()):
    raise KeyError(f"Primary metric {primary!r} not computed; available: {sorted(pd.DataFrame(rows).metric.unique())}")

df = pd.DataFrame(rows)
def sig4(x):
    return x if pd.isna(x) else float(f"{x:.4g}")
df["value"] = df["value"].map(sig4)
df["threshold"] = df["threshold"].map(sig4)
lower_better = {"brier", "log_loss", "mse", "mae"}
best = (min if primary in lower_better else max)(primary_vals, key=primary_vals.get)

mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics; binary threshold from Youden J")
for name, value in primary_vals.items():
    mlca_runtime.record_metric(primary, float(value), "val")
mlca_runtime.record_metric(primary, float(primary_vals[best]), "val")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(auc_vals[best]), source="validation_metrics.csv")
print(f"features={len(feature_cols)} models={len(models)} best={best} {primary}={primary_vals[best]:.4g}")
