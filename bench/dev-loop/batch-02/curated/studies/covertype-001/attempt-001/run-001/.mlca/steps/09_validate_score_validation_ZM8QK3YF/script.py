import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load("05_preprocess_fit_preprocessing_R1SQG5ET/preprocessor.pkl")
sel = load("06_features_select_features_830E7YS6/selector.pkl")
m1 = load("07_model_train_logistic_CM1VEX5J/model.pkl")
m2 = load("08_model_train_alternative_3YZTFRCC/alternative_model.pkl")

def input_cols(obj):
    if isinstance(obj, Pipeline):
        return input_cols(obj.steps[0][1])
    fn = getattr(obj, "feature_names_in_", None)
    return list(fn) if fn is not None else None

drop = list(targets) + list(excluded) + ([group] if group else [])
raw_cols = input_cols(pre)
if raw_cols is None:
    raw_cols = [c for c in train.columns if c not in drop]

def frontend():
    if input_cols(sel) == raw_cols:
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

def wrap(m):
    cols = input_cols(m)
    if cols is not None and all(c in val.columns for c in cols):
        return m
    return Pipeline([("frontend", frontend()), ("clf", m)])

def select_X(cols):
    if cols is not None and all(c in val.columns for c in cols):
        return val[cols]
    return val.drop(columns=drop, errors="ignore")

y_val = val[target].to_numpy()
rows = []
best_name, best_auc = None, -1.0
for name, m in [("logistic", m1), ("alternative", m2)]:
    mdl = wrap(m)
    X = select_X(input_cols(mdl))
    proba = mdl.predict_proba(X)
    classes = list(mdl.classes_)
    pred = mdl.predict(X)
    auc = float(roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes))
    Y = np.zeros((len(y_val), len(classes)))
    for i, c in enumerate(classes):
        Y[:, i] = (y_val == c).astype(float)
    pr = float(average_precision_score(Y, proba, average="macro"))
    ba = float(balanced_accuracy_score(y_val, pred))
    brier = float(np.mean((proba - Y) ** 2))
    for metric, value in [("roc_auc_ovr", auc), ("pr_auc_ovr", pr),
                          ("balanced_accuracy", ba), ("brier", brier)]:
        rows.append({"model": name, "metric": metric,
                     "value": float(f"{value:.4g}"), "threshold": np.nan})
    mlca_runtime.record_metric(f"roc_auc_{name}", auc, "val")
    if auc > best_auc:
        best_auc, best_name = auc, name

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(f"{best_auc:.4g}"), source="validation_metrics.csv")
print(len(df), "rows;", best_name, round(best_auc, 4))
