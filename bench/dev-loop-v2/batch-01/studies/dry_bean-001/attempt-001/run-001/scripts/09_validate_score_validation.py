import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
p = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in p:
        raise KeyError(f"Missing {k}; available: {list(p)}")
targets = p["targets"]
if len(targets) != 1:
    raise ValueError("Expected one target")
target = targets[0]
primary = p["task"]["primary_metric"]

val = pd.read_parquet(inp["val"])
val = val.loc[val[target].notna()]
y = val[target]

models = [
    ("logistic", pickle.load(open(inp["07_model_train_logistic_0BAYN51T/model.pkl"], "rb"))),
    ("alternative", pickle.load(open(inp["08_model_train_alternative_JA5Q5D9S/alternative_model.pkl"], "rb"))),
]

def classes_of(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    last = list(m.named_steps.values())[-1]
    return np.asarray(last.classes_)

def X_for(m):
    cols = getattr(m, "feature_names_in_", None)
    if cols is None:
        raise ValueError("model lacks feature_names_in_")
    return val.loc[:, list(cols)]

def eval_model(m):
    X = X_for(m)
    proba = m.predict_proba(X)
    classes = classes_of(m)
    present = [c for c in classes if c in set(y)]
    if len(present) == len(classes):
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        vals = [
            roc_auc_score((y == c).astype(int), proba[:, j])
            for j, c in enumerate(classes)
            if c in present
        ]
        auc = float(np.mean(vals))
        print("classes omitted:", [c for c in classes if c not in present])
    pred = classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y, pred, average="macro", labels=classes, zero_division=0)
    return {"roc_auc_ovr": float(auc), "f1_macro": float(f1)}

def sig4(v):
    return float(f"{v:.4g}") if v == v else v

lower = {"brier_score", "log_loss", "rmse", "mae"}
higher_better = primary not in lower
rows = []
per = {}
for name, m in models:
    met = eval_model(m)
    per[name] = met
    for metric, value in met.items():
        rows.append({"model": name, "metric": metric, "value": sig4(value), "threshold": np.nan})
    mlca_runtime.record_metric(primary, met[primary], "val")
    print(name, "primary", primary, met[primary])

best_name = (
    max(per, key=lambda n: per[n][primary])
    if higher_better
    else min(per, key=lambda n: per[n][primary])
)
best_primary = per[best_name][primary]
best_auc = per[best_name]["roc_auc_ovr"]
mlca_runtime.record_metric(primary, best_primary, "val")

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print("best", best_name, "val", primary, best_primary)
print("rows", len(metrics_df))
