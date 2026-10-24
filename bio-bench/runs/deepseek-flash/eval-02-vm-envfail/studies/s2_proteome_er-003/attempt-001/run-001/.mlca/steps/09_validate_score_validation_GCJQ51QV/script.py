import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = list(protocol["targets"])
target = targets[0]
group_col = protocol.get("group_column")
val = pd.read_parquet(inputs["val"])
y_val = val[target]

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_B1ZJVKDM/preprocessor.pkl")
sel = load_pkl("06_features_select_features_X6VS56A4/selector.pkl")
models = {
    "logistic": load_pkl("07_model_train_logistic_8VZRZDX0/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_Z005D6HS/alternative_model.pkl"),
}

def input_cols(obj):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is None and isinstance(obj, Pipeline):
        for _, step in obj.steps:
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                break
    return list(cols) if cols is not None else None

pre_cols, sel_cols = input_cols(pre), input_cols(sel)
raw_cols = pre_cols or sel_cols

def frontend():
    if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
        sc = input_cols(sel)
        if pre_cols and sc and set(sc) == set(pre_cols):
            return sel
    if pre is not None and sel is not None:
        return Pipeline([("preprocessor", pre), ("selector", sel)])
    return sel if sel is not None else pre

def candidate(obj):
    cols = input_cols(obj)
    if hasattr(obj, "predict_proba") and cols and all(c in val.columns for c in cols) and not any(c in targets for c in cols):
        return obj
    est = obj.steps[-1][1] if isinstance(obj, Pipeline) else obj
    front = frontend()
    if front is None:
        return est
    if isinstance(front, Pipeline):
        return Pipeline(list(front.steps) + [("model", est)])
    return Pipeline([("frontend", front), ("model", est)])

def compute(cand, X, y):
    classes = np.asarray(cand.classes_)
    proba = cand.predict_proba(X)
    if len(classes) == 2:
        pos = task.get("positive_label", classes[1])
        pos = pos if pos in classes else classes[1]
        j = int(np.where(classes == pos)[0][0])
        p = proba[:, j]
        yb = (y == pos).astype(int)
        auc = roc_auc_score(yb, p)
        ap = average_precision_score(yb, p)
        fpr, tpr, thr = roc_curve(yb, p)
        k = int(np.nanargmax(tpr - fpr))
        t = float(thr[k]) if np.isfinite(thr[k]) else float(np.max(p))
        ba = balanced_accuracy_score(yb, (p >= t).astype(int))
        brier = brier_score_loss(yb, p)
        return {"roc_auc": auc, "pr_auc": ap, "balanced_accuracy": ba, "brier_score": brier}, t
    pred = cand.predict(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        vals = []
        for j, c in enumerate(classes):
            if c in set(y):
                vals.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        auc = float(np.mean(vals)) if vals else np.nan
    f1 = f1_score(y, pred, average="macro", labels=classes, zero_division=0)
    return {"roc_auc_ovr": auc, "macro_f1": f1}, np.nan

rows, primary_vals = [], {}
higher_better = primary_metric not in {"brier_score", "log_loss"}
for name, obj in models.items():
    cand = candidate(obj)
    cols = input_cols(cand) or raw_cols
    if cols is None:
        excluded = set(protocol.get("excluded_columns", []))
        cols = [c for c in val.columns if c not in set(targets) and c not in excluded and c != group_col]
    if any(c in targets for c in cols):
        raise ValueError("Target column reached model input")
    X = val[list(cols)]
    m, t = compute(cand, X, y_val)
    for met, v in m.items():
        thr = t if (len(np.asarray(cand.classes_)) == 2 and met == "balanced_accuracy") else np.nan
        rows.append({"model": name, "metric": met, "value": v, "threshold": thr})
    primary_vals[name] = m.get(primary_metric, m.get("roc_auc"))

best_name = max(primary_vals, key=lambda n: primary_vals[n] if higher_better else -primary_vals[n])
best_val = float(primary_vals[best_name])
best_auc = next(r["value"] for r in rows if r["model"] == best_name and r["metric"] == "roc_auc")

df = pd.DataFrame(rows)
for col in ("value", "threshold"):
    df[col] = [float(f"{v:.4g}") if pd.notna(v) and np.isfinite(v) else v for v in df[col]]
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model; binary threshold is Youden J on validation.")
mlca_runtime.record_metric(primary_metric, best_val, split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")
print(f"val_rows={len(val)} candidates={len(models)}")
print(f"best_model={best_name}")
print(f"best_val_roc_auc={best_auc:.4g}")
print("wrote validation_metrics.csv")
