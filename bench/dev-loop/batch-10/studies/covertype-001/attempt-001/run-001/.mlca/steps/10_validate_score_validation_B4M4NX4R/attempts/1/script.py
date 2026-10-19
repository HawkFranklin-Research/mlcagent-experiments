import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group_col = proto.get("group_column")
metric = task["primary_metric"]

tr = pd.read_parquet(inp["train"])
va = pd.read_parquet(inp["val"])
y_tr = tr[targets[0]]
y_va = va[targets[0]]

feat_cols = [c for c in tr.columns if c not in targets and c not in excluded and c != group_col]
keep = []
for c in feat_cols:
    s = tr[c]
    if s.dtype == object:
        if s.nunique() == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            continue
    keep.append(c)
feat_cols = keep
X_tr = tr[feat_cols]
X_va = va[feat_cols]

models = {}
with open(inp["08_model_train_logistic_6YJ0NC7Y/model.pkl"], "rb") as f:
    models["logistic"] = pickle.load(f)
with open(inp["09_model_train_alternative_143BSS6M/alternative_model.pkl"], "rb") as f:
    models["alternative"] = pickle.load(f)

def make_X(model, X):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X[cols]
    return X

rows = []
best_name, best_val = None, -np.inf
for name, m in models.items():
    Xv = make_X(m, X_va)
    proba = m.predict_proba(Xv)
    classes = list(m.classes_)
    present = [c for c in classes if c in set(y_va.unique())]
    idx = [i for i, c in enumerate(classes) if c in present]
    try:
        auc = roc_auc_score(y_va, proba[:, idx], multi_class="ovr", average="macro", labels=classes[idx])
    except Exception:
        vals = []
        for c in present:
            j = classes.index(c)
            vals.append(roc_auc_score((y_va == c).astype(int), proba[:, j].sum(axis=1) if proba.ndim == 3 else proba[:, j]))
        auc = float(np.mean(vals))
    f1 = f1_score(y_va, m.predict(Xv), average="macro")
    rows.append({"model": name, "metric": metric, "value": round(float(auc), 6), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": round(float(f1), 6), "threshold": np.nan})
    mlca_runtime.record_metric(f"{name}_val_{metric}", float(auc), split="val")
    if auc > best_val:
        best_val, best_name = float(auc), name

tbl = pd.DataFrame(rows)
mlca_runtime.save_table(tbl, "validation_metrics.csv", caption="Validation metrics per model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", round(best_val, 6), source="validation_metrics.csv")
print(f"rows={len(tbl)} models={len(models)} metric={metric}")
print(f"best={best_name} val_{metric}={best_val:.4f}")
