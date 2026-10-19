import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve, f1_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d)}")
    return d[k]

task = req(protocol, "task")
primary = req(task, "primary_metric")
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError("This validation script expects one target")
target = targets[0]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
forbidden = set(targets) | set(excluded)
if group:
    forbidden.add(group)

def keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False

feature_cols = [c for c in train.columns if c not in forbidden and not keylike(train[c])]
model_keys = {
    "logistic": "07_model_train_logistic_ZJV6JEEV/model.pkl",
    "alternative": "08_model_train_alternative_YVXTQSN7/alternative_model.pkl",
}
models = {}
for name, key in model_keys.items():
    with open(paths[key], "rb") as f:
        models[name] = pickle.load(f)

pos_declared = task.get("positive_label", task.get("positive_class", None))
y = val[target].to_numpy()
records = []
primary_vals = {}

def sig4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(f"{float(x):.4g}")

for name, model in models.items():
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
    if any(c in forbidden for c in cols):
        raise ValueError("Target or excluded column in model input")
    X = val[cols]
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{name} has no predict_proba")
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    n_classes = task.get("n_classes", len(classes))
    if n_classes > 2 or len(classes) > 2:
        present = [c for c in classes if np.any(y == c)]
        if len(present) == len(classes):
            auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            auc = np.mean([roc_auc_score((y == c).astype(int), proba[:, j])
                           for j, c in enumerate(classes) if c in present])
            print("classes left out:", sorted(set(classes) - set(present)))
        pred = model.predict(X)
        f1 = f1_score(y, pred, average="macro", labels=classes)
        records += [{"model": name, "metric": "roc_auc", "value": sig4(auc), "threshold": np.nan},
                    {"model": name, "metric": "f1", "value": sig4(f1), "threshold": np.nan}]
        primary_vals[name] = auc if primary in ("roc_auc", "roc_auc_ovr") else f1 if primary == "f1" else np.nan
    else:
        positive = pos_declared
        if positive is None:
            positive = classes[1]
        if positive not in classes:
            for cast in (int, str, float):
                try:
                    candidate = cast(positive)
                    if candidate in classes:
                        positive = candidate
                        break
                except Exception:
                    pass
        if positive not in classes:
            raise ValueError(f"Positive label {positive} not in model classes {classes}")
        j = classes.index(positive)
        proba_pos = proba[:, j]
        y_bin = (y == positive).astype(int)
        fpr, tpr, thresholds = roc_curve(y_bin, proba_pos)
        best_thr = float(thresholds[np.argmax(tpr - fpr)])
        pred = (proba_pos >= best_thr).astype(int)
        vals = {
            "roc_auc": roc_auc_score(y_bin, proba_pos),
            "pr_auc": average_precision_score(y_bin, proba_pos),
            "balanced_accuracy": balanced_accuracy_score(y_bin, pred),
            "brier": brier_score_loss(y_bin, proba_pos),
        }
        for metric, value in vals.items():
            records.append({"model": name, "metric": metric, "value": sig4(value), "threshold": sig4(best_thr)})
        primary_vals[name] = vals.get(primary, np.nan)

direction = {"brier": -1, "log_loss": -1, "rmse": -1, "mae": -1, "mse": -1}
higher_better = direction.get(primary, 1) > 0
best = max(primary_vals, key=lambda k: primary_vals[k]) if higher_better else min(primary_vals, key=lambda k: primary_vals[k])
best_auc = next(r["value"] for r in records if r["model"] == best and r["metric"] == "roc_auc")
df = pd.DataFrame(records, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by model.")
for name in [m for m in primary_vals if m != best] + [best]:
    mlca_runtime.record_metric(primary, float(primary_vals[name]), "val")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")
print(len(feature_cols), len(val))
print(best, primary, float(primary_vals[best]))
print(best_auc)
