import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = task["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
y_val = val[target].values

model_keys = {
    "logistic": "07_model_train_logistic_E9C4BBRQ/model.pkl",
    "alternative": "08_model_train_alternative_8T0DW9QP/alternative_model.pkl",
}

def load_model(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def predict_proba(m, df):
    cols = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else feature_cols
    return m.predict_proba(df[cols]), np.array(m.classes_)

rows = []
best, best_auc = None, -np.inf
for name, key in model_keys.items():
    m = load_model(inputs[key])
    proba, classes = predict_proba(m, val)
    aucs = []
    for j, c in enumerate(classes):
        yb = (y_val == c).astype(int)
        if 0 < yb.sum() < len(yb):
            aucs.append(roc_auc_score(yb, proba[:, j]))
    macro_auc = float(np.mean(aucs))
    preds = classes[np.argmax(proba, axis=1)]
    macro_f1 = float(f1_score(y_val, preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_val, preds))
    Y = np.zeros_like(proba)
    for j, c in enumerate(classes):
        Y[:, j] = (y_val == c).astype(int)
    brier = float(np.mean(np.sum((proba - Y) ** 2, axis=1)))
    for mname, mval in [("roc_auc_ovr", macro_auc), ("macro_f1", macro_f1),
                        ("balanced_accuracy", bal_acc), ("brier", brier)]:
        rows.append({"model": name, "metric": mname,
                     "value": float(f"{mval:.4g}"), "threshold": np.nan})
    mlca_runtime.record_metric(f"{primary}__{name}", macro_auc, "val")
    if macro_auc > best_auc:
        best, best_auc = name, macro_auc

mlca_runtime.save_table(pd.DataFrame(rows), "validation_metrics.csv",
                        caption="Validation metrics per candidate (multiclass, no binary threshold)")
mlca_runtime.record_metric(primary, best_auc, "val")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(f"models: {list(model_keys)}  val_rows: {len(val)}  features: {len(feature_cols)}")
print(f"{primary} best={best} val={best_auc:.4f}")
