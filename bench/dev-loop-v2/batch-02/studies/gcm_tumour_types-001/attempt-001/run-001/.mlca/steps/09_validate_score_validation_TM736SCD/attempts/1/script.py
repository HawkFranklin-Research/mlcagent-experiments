import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "task" not in proto or "primary_metric" not in proto["task"]:
    raise ValueError(f"missing task.primary_metric; top-level keys: {list(proto)}")
primary = proto["task"]["primary_metric"]
if primary != "roc_auc_ovr":
    raise ValueError(f"declared primary_metric {primary!r} is not handled by this multiclass validator")
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
if target not in val.columns or target not in train.columns:
    raise ValueError(f"target {target!r} not present in train/val")

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=False) == len(s):
        continue
    lc = str(c).lower()
    if any(t in lc for t in ("row_id", "rowid", "index", "idx", "_id", "id_")):
        continue
    feat_cols.append(c)

X_val = val[feat_cols].copy()
y_val = val[target].to_numpy()

def load(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load("05_preprocess_fit_preprocessing_DXVQHGBE/preprocessor.pkl")
sel = load("06_features_select_features_9JJFXST3/selector.pkl")
models = [
    ("logistic", load("07_model_train_logistic_6BHE1H1F/model.pkl")),
    ("alternative", load("08_model_train_alternative_SGR2R7DJ/alternative_model.pkl")),
]

def names_in(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is None and isinstance(obj, Pipeline) and obj.steps:
        names = getattr(obj.steps[0][1], "feature_names_in_", None)
    try:
        return list(names) if names is not None else None
    except Exception:
        return None

raw_set = set(feat_cols)
def accepts_raw(obj):
    n = names_in(obj)
    return n is not None and all(x in raw_set for x in n)

def candidates(model):
    if pre is not None and sel is not None and not accepts_raw(sel):
        yield "preprocess+selector+model", Pipeline([("preprocess", pre), ("selector", sel), ("model", model)])
    if sel is not None:
        yield "selector+model", Pipeline([("selector", sel), ("model", model)])
    if pre is not None:
        yield "preprocess+model", Pipeline([("preprocess", pre), ("model", model)])
    yield "direct", model

def predict_candidate(est):
    n = names_in(est)
    Xs = X_val[list(n)] if n is not None else X_val
    proba = est.predict_proba(Xs)
    pred = est.predict(Xs)
    classes = list(est.classes_)
    if proba.shape[1] != len(classes):
        raise ValueError("probability matrix and classes_ are misaligned")
    return proba, pred, classes

def macro_ovr(y, proba, classes):
    present = [i for i, c in enumerate(classes) if np.any(y == c)]
    omitted = [c for i, c in enumerate(classes) if i not in present]
    vals = []
    for i in present:
        yb = (y == classes[i]).astype(int)
        if len(np.unique(yb)) > 1:
            vals.append(roc_auc_score(yb, proba[:, i]))
    return (float(np.mean(vals)) if vals else float("nan")), omitted

rows = []
best_name, best_auc, best_omitted = None, -np.inf, []
for name, model in models:
    last_err = None
    for wrap, est in candidates(model):
        try:
            proba, pred, classes = predict_candidate(est)
            auc, omitted = macro_ovr(y_val, proba, classes)
            f1 = f1_score(y_val, pred, average="macro", labels=classes, zero_division=0)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"could not evaluate {name}: {last_err}")

    rows.append({"model": name, "metric": primary, "value": float(f"{auc:.4g}"), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f"{f1:.4g}"), "threshold": np.nan})
    mlca_runtime.record_metric(f"{primary}__{name}", float(auc), split="val")
    if auc > best_auc:
        best_name, best_auc, best_omitted = name, float(auc), omitted

if best_name is None:
    raise RuntimeError("no candidate produced a finite primary metric")

mlca_runtime.save_table(pd.DataFrame(rows), "validation_metrics.csv", caption="Validation metrics for declared multiclass primary metric.")
mlca_runtime.record_metric(primary, best_auc, split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"primary={primary} models={len(models)} best={best_name}")
print(f"best_val_roc_auc={best_auc:.4f}")
if best_omitted:
    print("omitted_val_classes=" + ",".join(map(str, best_omitted)))
