import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    balanced_accuracy_score, brier_score_loss, f1_score
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs["_protocol"], "r") as f:
    proto = json.load(f)

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available keys: {list(d)}")
    return d[k]

task = need(proto, "task")
primary = need(task, "primary_metric")
targets = list(need(proto, "targets"))
exc = need(proto, "excluded_columns")
if isinstance(exc, dict):
    exc = exc.get("columns", exc.get("examples", []))
exc = list(exc)
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

key_cols = []
for c in train.columns:
    if c in targets or c in exc or c == group:
        continue
    col = train[c]
    if col.nunique(dropna=False) != len(train):
        continue
    if (pd.api.types.is_object_dtype(col) or
        pd.api.types.is_string_dtype(col) or
        isinstance(col.dtype, pd.CategoricalDtype) or
        pd.api.types.is_integer_dtype(col)):
        key_cols.append(c)

features = [
    c for c in train.columns
    if c not in set(targets + exc + ([group] if group else []) + key_cols)
]

model_paths = {
    "logistic": inputs["07_model_train_logistic_4PCHXMGW/model.pkl"],
    "alternative": inputs["08_model_train_alternative_G2SMAQN6/alternative_model.pkl"],
}

def get_X(model, frame):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        cols = features
    return frame[list(cols)]

rows = []
primary_values = {}
roc_values = {}

for name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    X = get_X(model, val)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    y = val[targets[0]]

    if len(classes) == 2:
        pos_label = task.get("positive_label", classes[1])
        j = classes.index(pos_label) if pos_label in classes else 1
        p = proba[:, j]
        y_bin = (y == pos_label).astype(int)
        roc = roc_auc_score(y_bin, p)
        pr = average_precision_score(y_bin, p)
        fpr, tpr, thr = roc_curve(y_bin, p)
        k = int(np.argmax(tpr - fpr))
        th = float(thr[k])
        bal = balanced_accuracy_score(y_bin, (p >= th).astype(int))
        brier = brier_score_loss(y_bin, p)
        for metric, value, threshold in [
            ("roc_auc", roc, np.nan),
            ("pr_auc", pr, np.nan),
            ("balanced_accuracy", bal, th),
            ("brier_score", brier, np.nan),
        ]:
            rows.append({"model": name, "metric": metric, "value": value, "threshold": threshold})
        primary_values[name] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "balanced_accuracy": bal,
            "brier_score": brier,
        }[primary]
        roc_values[name] = roc
        mlca_runtime.record_metric(primary, primary_values[name], "val")
    else:
        present = [c for c in classes if (y == c).any()]
        vals = []
        for c in present:
            j = classes.index(c)
            vals.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        macro_auc = float(np.mean(vals))
        y_pred = model.predict(X)
        macro_f1 = f1_score(y, y_pred, average="macro")
        rows.append({"model": name, "metric": "roc_auc", "value": macro_auc, "threshold": np.nan})
        rows.append({"model": name, "metric": "f1", "value": macro_f1, "threshold": np.nan})
        primary_values[name] = macro_auc if primary == "roc_auc" else macro_f1
        roc_values[name] = macro_auc
        mlca_runtime.record_metric(primary, primary_values[name], "val")

direction = {"roc_auc": 1, "pr_auc": 1, "balanced_accuracy": 1, "brier_score": -1, "f1": 1}
sign = direction.get(primary, 1)
best_name = max(primary_values, key=primary_values.get) if sign == 1 else min(primary_values, key=primary_values.get)
best_auc = roc_values[best_name]

def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
df["value"] = df["value"].apply(sig4)
df["threshold"] = df["threshold"].apply(sig4)

mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(sig4(best_auc)), source="validation_metrics.csv")

print(f"val shape={val.shape}; features={len(features)}")
print(f"best_model={best_name}; best_val_roc_auc={sig4(best_auc)}")
