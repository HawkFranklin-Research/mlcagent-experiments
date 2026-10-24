import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for k in ["task", "targets", "excluded_columns"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if s.is_unique and s.dtype.kind in ("O", "U", "S", "i", "u"):
        return True
    return False

features = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if is_key_like(train[c]):
        continue
    features.append(c)

def get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def get_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if isinstance(model, Pipeline):
        return list(model.steps[-1][1].classes_)
    raise AttributeError("Model missing classes_")

model_paths = {
    "logistic": inp["07_model_train_logistic_5W0X2YFF/model.pkl"],
    "alternative": inp["08_model_train_alternative_T7JN5GQ1/alternative_model.pkl"],
}
models = {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

target = targets[0]
val_y = val[target]

results = {}
for name, model in models.items():
    fnames = get_feature_names(model) or features
    bad = [c for c in fnames if c in targets or c in excluded]
    if bad:
        raise ValueError(f"Model {name} expects excluded/target columns: {bad}")
    Xv = val[fnames]
    proba = model.predict_proba(Xv)
    classes = get_classes(model)
    if proba.shape[1] != len(classes):
        raise ValueError(f"Proba shape {proba.shape} != classes {len(classes)}")
    present = [c for c in classes if c in val_y.unique()]
    aucs = []
    for c in present:
        j = classes.index(c)
        aucs.append(roc_auc_score((val_y == c).astype(int), proba[:, j]))
    auc_val = float(np.mean(aucs)) if aucs else float("nan")
    pred = model.predict(Xv)
    f1_val = float(f1_score(val_y, pred, average="macro", labels=classes, zero_division=0))
    results[name] = {"auc": auc_val, "f1": f1_val}

rows = []
for name, res in results.items():
    rows.append({"model": name, "metric": primary_metric, "value": res["auc"], "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": res["f1"], "threshold": np.nan})
df = pd.DataFrame(rows)
df["value"] = df["value"].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models")

for name, res in results.items():
    mlca_runtime.record_metric(f"{name}_{primary_metric}", res["auc"], "val")
best_name = max(results, key=lambda k: results[k]["auc"])
best_auc = results[best_name]["auc"]
mlca_runtime.record_metric(primary_metric, best_auc, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"val rows: {len(val)}")
print(f"best model: {best_name}")
print(f"best val {primary_metric}: {best_auc:.4f}")
