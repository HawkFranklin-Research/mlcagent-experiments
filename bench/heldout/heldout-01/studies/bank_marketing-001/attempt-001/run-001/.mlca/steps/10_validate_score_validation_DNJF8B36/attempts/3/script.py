import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(protocol.keys())}")
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
drop = set(targets) | excluded | ({group} if group else set())

def key_like(s):
    if pd.api.types.is_float_dtype(s):
        return False
    if s.nunique(dropna=False) != len(s):
        return False
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s)

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
X_val = val[feature_cols].copy()
y_raw = val[target].to_numpy()

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"])
sel = load(inp["06_features_select_features_TX9DR60B/selector.pkl"])
models = [
    ("logistic", load(inp["07_model_train_logistic_GD4X7DTC/model.pkl"])),
    ("alternative", load(inp["08_model_train_alternative_0A2EAXQK/alternative_model.pkl"])),
]

def classes_of(obj):
    o = obj
    while hasattr(o, "steps") and o.steps:
        o = o.steps[-1][1]
    return getattr(o, "classes_", None)

def usable(obj, X):
    names = getattr(obj, "feature_names_in_", None)
    if names is None:
        return X
    names = list(names)
    if not all(n in X.columns for n in names):
        raise ValueError("model expects columns absent from the frame")
    return X[names]

def predict_any(model_obj, X):
    cands = [model_obj]
    if sel is not None:
        cands.append(Pipeline([("selector", sel), ("model", model_obj)]))
    if pre is not None and sel is not None:
        cands.append(Pipeline([("preprocessor", pre), ("selector", sel), ("model", model_obj)]))
    if pre is not None:
        cands.append(Pipeline([("preprocessor", pre), ("model", model_obj)]))
    errs = []
    for obj in cands:
        try:
            pr = np.asarray(obj.predict_proba(usable(obj, X)), dtype=float)
            if pr.ndim != 2 or pr.shape[0] != len(X) or not np.isfinite(pr).all():
                raise ValueError("malformed probability output")
            return obj, pr
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    raise RuntimeError("No working pipeline: " + " | ".join(errs))

higher = {"roc_auc", "pr_auc", "balanced_accuracy", "f1", "f1_macro", "accuracy",
          "roc_auc_ovr", "average_precision", "r2"}
lower = {"brier", "log_loss", "mae", "mse", "rmse"}
if primary not in higher and primary not in lower:
    raise ValueError(f"Unknown primary metric direction: {primary}")

def align_y(y, classes):
    cls = list(classes)
    yv = pd.Series(y)
    if set(pd.unique(yv.dropna())) <= set(cls):
        return yv.to_numpy()
    return yv.astype(str).map({str(c): c for c in cls}).to_numpy()

declared_pos = task.get("positive_label", task.get("positive_class", None))
rows, primary_vals, roc_vals = [], {}, {}
for name, obj in models:
    fitted, proba = predict_any(obj, X_val)
    cls = classes_of(fitted)
    if cls is None or len(cls) != 2:
        raise ValueError(f"{name}: expected two model classes, got {cls}")
    cls = np.asarray(cls)
    y_al = align_y(y_raw, cls)
    if pd.isna(pd.Series(y_al)).any():
        raise ValueError(f"{name}: cannot align target values to classes {cls.tolist()}")
    if declared_pos is None:
        pos = cls[1]
    else:
        pos = declared_pos if declared_pos in cls else {str(c): c for c in cls}.get(str(declared_pos), cls[1])
    j = int(np.where(cls == pos)[0][0])
    p = np.clip(proba[:, j], 0.0, 1.0)
    y_bin = (y_al == pos).astype(int)
    if y_bin.min() == y_bin.max():
        raise ValueError(f"{name}: single-class binary indicator after alignment")
    roc = float(roc_auc_score(y_bin, p))
    pr = float(average_precision_score(y_bin, p))
    brier = float(brier_score_loss(y_bin, p))
    fpr, tpr, thr = roc_curve(y_bin, p)
    jj = int(np.argmax(np.where(np.isfinite(thr), tpr - fpr, -np.inf)))
    youden = float(thr[jj]) if np.isfinite(thr[jj]) else np.nan
    pred = (p >= youden).astype(int) if np.isfinite(youden) else (p >= 0.5).astype(int)
    bal = float(balanced_accuracy_score(y_bin, pred))
    vals = {"roc_auc": (roc, np.nan), "pr_auc": (pr, np.nan),
            "balanced_accuracy": (bal, youden), "brier": (brier, np.nan)}
    if primary not in vals:
        raise ValueError(f"Primary metric {primary} not computable here")
    prim = float(vals[primary][0])
    if not np.isfinite(prim):
        raise ValueError(f"{name}: non-finite {primary}")
    primary_vals[name] = prim
    roc_vals[name] = roc
    mlca_runtime.record_metric(primary, prim, "val")
    ordered = [(primary, vals[primary])] + [(m, v) for m, v in vals.items() if m != primary]
    for m, (v, t) in ordered:
        rows.append({"model": name, "metric": m, "value": v, "threshold": t})

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
for c in ("value", "threshold"):
    df[c] = df[c].map(lambda x: np.nan if pd.isna(x) else float(f"{float(x):.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv",
                        caption="Validation metrics for candidate models.")

ascending = primary in lower
best_model = min(primary_vals, key=primary_vals.get) if ascending else max(primary_vals, key=primary_vals.get)
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(roc_vals[best_model]),
                                   source="validation_metrics.csv")
print(f"features={len(feature_cols)} train={train.shape} val={val.shape} primary={primary}")
print(f"best_model={best_model} {primary}={primary_vals[best_model]:.4g} roc_auc={roc_vals[best_model]:.4g}")
