import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
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
protocol = json.loads(inp["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing protocol['targets']; available top-level keys: {list(protocol)}")
if "task" not in protocol:
    raise KeyError(f"Missing protocol['task']; available top-level keys: {list(protocol)}")

targets = protocol["targets"]
task = protocol["task"]
primary_metric = task["primary_metric"]
group_col = protocol.get("group_column")
ex_raw = protocol.get("excluded_columns", [])
if isinstance(ex_raw, dict):
    ex_raw = ex_raw.get("columns") or ex_raw.get("examples") or []
excluded = set(ex_raw)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = targets[0]
if target not in val.columns:
    raise KeyError(f"Target {target!r} not in validation columns")

def is_keylike(s):
    if pd.api.types.is_numeric_dtype(s):
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s) and s.notna().all()

feature_cols = [
    c for c in train.columns
    if c not in set(targets) and c not in excluded and c != group_col and not is_keylike(train[c])
]

def load_pkl(key):
    with inp[key].open("rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("06_preprocess_fit_preprocessing_VD4Z4Z0N/preprocessor.pkl")
selector = load_pkl("07_features_select_features_AKBM21YH/selector.pkl")

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return list(obj.classes_)
    if hasattr(obj, "named_steps"):
        return get_classes(list(obj.named_steps.values())[-1])
    if hasattr(obj, "steps"):
        return get_classes(obj.steps[-1][1])
    return None

def transform_frontend(df):
    if selector is not None:
        if isinstance(selector, Pipeline) and getattr(selector, "feature_names_in_", None) is not None:
            scols = list(selector.feature_names_in_)
            if all(c in df.columns for c in scols):
                return selector.transform(df[scols])
        if preprocessor is not None:
            pcols = list(getattr(preprocessor, "feature_names_in_", feature_cols))
            xpre = preprocessor.transform(df[pcols])
        else:
            xpre = df[feature_cols]
        return selector.transform(xpre)
    if preprocessor is not None:
        pcols = list(getattr(preprocessor, "feature_names_in_", feature_cols))
        return preprocessor.transform(df[pcols])
    return df[feature_cols]

def predict_proba(model, df):
    if isinstance(model, Pipeline):
        cols = getattr(model, "feature_names_in_", None)
        x = df[list(cols)] if cols is not None else df[feature_cols]
        return model.predict_proba(x)
    if selector is None and preprocessor is None:
        cols = list(getattr(model, "feature_names_in_", feature_cols))
        return model.predict_proba(df[cols])
    x = transform_frontend(df)
    mcols = getattr(model, "feature_names_in_", None)
    if mcols is not None and isinstance(x, pd.DataFrame):
        x = x[list(mcols)]
    return model.predict_proba(x)

def sig4(x):
    try:
        if x is None or not np.isfinite(float(x)):
            return np.nan
        return float(f"{float(x):.4g}")
    except Exception:
        return x

candidate_keys = [
    "08_model_train_logistic_YSEARS1W/model.pkl",
    "09_model_train_alternative_P29JT10K/alternative_model.pkl",
    "latest_model",
]
cands = {}
for key in candidate_keys:
    if key in inp:
        rp = inp[key].resolve()
        if rp not in cands:
            cands[rp] = (key, load_pkl(key))

def cand_name(key):
    if "logistic" in key:
        return "logistic"
    if "alternative" in key:
        return "alternative"
    if key == "latest_model":
        return "latest_model"
    return key.split("/")[0]

df_eval = val.loc[val[target].notna()]
y = df_eval[target]
declared_pos = task.get("positive_label") or task.get("positive_class") or protocol.get("positive_label")
direction = 1 if primary_metric.lower() in {"roc_auc", "pr_auc", "balanced_accuracy", "accuracy", "f1", "f1_macro"} else -1
rows, best_name, best_val, auc_by_name = [], None, None, {}

for rp, (key, model) in cands.items():
    name = cand_name(key)
    proba = predict_proba(model, df_eval)
    classes = get_classes(model)
    if classes is None:
        classes = list(pd.unique(y))
    pos_label = declared_pos if declared_pos is not None else (classes[1] if len(classes) > 1 else classes[0])
    if pos_label not in classes:
        raise ValueError(f"Positive label {pos_label!r} not in model classes {classes}")
    pos_idx = classes.index(pos_label)
    p = proba[:, pos_idx] if getattr(proba, "ndim", 1) > 1 else proba
    y_bin = (y == pos_label).to_numpy(dtype=int)

    roc = float(roc_auc_score(y_bin, p))
    pr = float(average_precision_score(y_bin, p))
    brier = float(brier_score_loss(y_bin, p))
    fpr, tpr, thresholds = roc_curve(y_bin, p)
    th = float(thresholds[int(np.argmax(tpr - fpr))])
    bal = float(balanced_accuracy_score(y_bin, (p >= th).astype(int)))

    metrics = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier": brier}
    pm = primary_metric.lower()
    if pm in metrics:
        val_primary = metrics[pm]
    elif pm == "roc_auc_score":
        val_primary = roc
    elif pm in {"average_precision", "pr_auc"}:
        val_primary = pr
    else:
        raise KeyError(f"Unsupported primary metric {primary_metric!r}")

    mlca_runtime.record_metric(primary_metric, float(val_primary), split="val")
    for mname, mval in metrics.items():
        rows.append({"model": name, "metric": mname, "value": sig4(mval), "threshold": sig4(th)})

    auc_by_name[name] = roc
    if best_val is None or direction * val_primary > direction * best_val:
        best_val, best_name = val_primary, name

df_metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(df_metrics, "validation_metrics.csv", caption="Validation metrics; binary threshold from Youden J on validation.")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", sig4(auc_by_name[best_name]), source="validation_metrics.csv")

print(f"val={len(df_eval)} features={len(feature_cols)} candidates={len(cands)}")
print(f"primary={primary_metric} best={best_name} value={sig4(best_val)} pos_label={pos_label!r}")
print(f"wrote validation_metrics.csv rows={len(df_metrics)}")
