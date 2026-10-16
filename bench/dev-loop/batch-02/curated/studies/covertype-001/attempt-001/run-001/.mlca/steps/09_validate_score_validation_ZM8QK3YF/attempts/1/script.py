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
task = proto["task"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_R1SQG5ET/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_830E7YS6/selector.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inputs["07_model_train_logistic_CM1VEX5J/model.pkl"], "rb") as f:
    m1 = pickle.load(f)
with open(inputs["08_model_train_alternative_3YZTFRCC/alternative_model.pkl"], "rb") as f:
    m2 = pickle.load(f)

def frontend():
    if isinstance(sel, Pipeline):
        names = [n.lower() for n in sel.named_steps]
        if any("preprocess" in n for n in names):
            return sel
    return Pipeline([("pre", pre), ("sel", sel)])

def wrap(m):
    if isinstance(m, Pipeline):
        names = [n.lower() for n in m.named_steps]
        if any("preprocess" in n or "select" in n for n in names):
            return m
    return Pipeline([("frontend", frontend()), ("clf", m)])

models = {"logistic": wrap(m1), "alternative": wrap(m2)}

drop = list(targets) + list(excluded) + ([group] if group else [])
X_val = val.drop(columns=drop, errors="ignore")
y_val = val[target].values

rows = []
best_name, best_auc = None, -1.0
for name, mdl in models.items():
    proba = mdl.predict_proba(X_val)
    classes = list(mdl.classes_)
    pred = mdl.predict(X_val)
    auc = float(roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes))
    pr = float(average_precision_score(pd.get_dummies(y_val).reindex(columns=classes, fill_value=0).values,
                                       proba, average="macro"))
    ba = float(balanced_accuracy_score(y_val, pred))
    Y = pd.get_dummies(y_val).reindex(columns=classes, fill_value=0).values
    brier = float(np.mean((proba - Y) ** 2))
    for metric, value in [("roc_auc_ovr", auc), ("pr_auc_ovr", pr),
                          ("balanced_accuracy", ba), ("brier", brier)]:
        rows.append({"model": name, "metric": metric, "value": value, "threshold": np.nan})
    mlca_runtime.record_metric(f"roc_auc_{name}", auc, "val")
    if auc > best_auc:
        best_auc, best_name = auc, name

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(len(df), "rows;", best_name, round(best_auc, 4))
