import mlca_runtime
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_curve,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())
task = protocol["task"]
primary_name = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets + excluded + ([group_col] if group_col else []))
feature_cols = [c for c in train.columns if c not in drop]
key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s) and c.lower().endswith("id"):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_F9B0NJ26/preprocessor.pkl")
sel = load_pkl("06_features_select_features_AA6KHSXZ/selector.pkl")
model1 = load_pkl("07_model_train_logistic_23SCN2HD/model.pkl")
model2 = load_pkl("08_model_train_alternative_9VT6FVM1/alternative_model.pkl")

def names(obj):
    if hasattr(obj, "feature_names_in_") and obj.feature_names_in_ is not None:
        return [str(x) for x in obj.feature_names_in_]
    return None

raw_set = set(feature_cols)

def frontend_for(pre_obj, sel_obj):
    sn = names(sel_obj) if sel_obj is not None else None
    pn = names(pre_obj) if pre_obj is not None else None
    if sel_obj is not None and sn is not None and set(sn).issubset(raw_set):
        return sel_obj
    if sel_obj is not None and pre_obj is not None and pn is not None and set(pn).issubset(raw_set):
        if sn is None or not set(sn).issubset(raw_set):
            return Pipeline([("preprocessor", pre_obj), ("selector", sel_obj)])
    if sel_obj is not None and sn is None:
        return sel_obj
    if pre_obj is not None and (pn is None or set(pn).issubset(raw_set)):
        return pre_obj
    return None

front = frontend_for(pre, sel)

def candidate_for(model_obj):
    mn = names(model_obj)
    if isinstance(model_obj, Pipeline):
        if mn is None or set(mn).issubset(raw_set):
            return model_obj
    elif mn is not None and set(mn).issubset(raw_set):
        return model_obj
    if front is not None:
        return Pipeline([("frontend", front), ("model", model_obj)])
    return model_obj

def get_X(model, df):
    mn = names(model)
    if mn is not None:
        missing = [c for c in mn if c not in df.columns]
        if missing:
            raise ValueError(f"missing model input columns: {missing}")
        return df[mn]
    return df[feature_cols]

def evaluate(name, model_obj):
    model = candidate_for(model_obj)
    X = get_X(model, val)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pos = classes[1]
    p = proba[:, classes.index(pos)]
    y = val[targets[0]]
    y_bin = (y == pos).astype(int).to_numpy()
    auc = roc_auc_score(y_bin, p)
    ap = average_precision_score(y_bin, p)
    fpr, tpr, thr = roc_curve(y_bin, p)
    j = tpr - fpr
    finite = np.isfinite(thr)
    threshold = float(thr[finite][np.argmax(j[finite])]) if finite.any() else 0.5
    y_pred = (p >= threshold).astype(int)
    ba = balanced_accuracy_score(y_bin, y_pred)
    brier = brier_score_loss(y_bin, p)
    return {
        "model": name,
        "roc_auc": auc,
        "pr_auc": ap,
        "balanced_accuracy": ba,
        "brier_score": brier,
        "threshold": threshold,
    }

results = [evaluate("logistic", model1), evaluate("alternative", model2)]
rows = []
for r in results:
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]:
        rows.append({
            "model": r["model"],
            "metric": metric,
            "value": r[metric],
            "threshold": r["threshold"],
        })
    mlca_runtime.record_metric(primary_name, r["roc_auc"], split="val")

metrics_df = pd.DataFrame(rows)
for c in ["value", "threshold"]:
    metrics_df[c] = metrics_df[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics for candidate models")

higher_better = primary_name not in {"brier_score", "log_loss", "rmse", "mae"}
best = max(results, key=lambda r: r["roc_auc"]) if higher_better else min(results, key=lambda r: r["roc_auc"])
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best["roc_auc"], source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("features", len(feature_cols))
print("best_model", best["model"])
print("best_val_roc_auc", best["roc_auc"])
