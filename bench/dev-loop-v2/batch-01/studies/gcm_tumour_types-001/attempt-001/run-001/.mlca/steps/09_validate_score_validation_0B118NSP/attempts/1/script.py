import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}. Available top-level keys: {list(protocol.keys())}")
primary = task.get("primary_metric")
if not primary:
    raise KeyError(f"Missing task.primary_metric. Available top-level keys: {list(protocol.keys())}")
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s):
    if s.dtype.kind in ("O", "U", "S") or str(s.dtype) == "category":
        return s.nunique(dropna=False) == len(s)
    if s.dtype.kind in ("i", "u"):
        return s.nunique(dropna=False) == len(s)
    return False

drop = set(targets + excluded + ([group] if group else []))
features = [c for c in train.columns if c not in drop and not key_like(train[c])]

def load_obj(key):
    with open(inputs[key], "rb") as f:
        obj = pickle.load(f)
    return obj.get("model", obj) if isinstance(obj, dict) else obj

pre = load_obj("05_preprocess_fit_preprocessing_63H1R042/preprocessor.pkl")
sel = load_obj("06_features_select_features_0WMEZ5WE/selector.pkl")
models = [
    ("logistic", load_obj("07_model_train_logistic_FX65AH12/model.pkl")),
    ("alternative", load_obj("08_model_train_alternative_R7D6ZBXG/alternative_model.pkl")),
]

def in_cols(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

def out_cols(obj):
    try:
        return list(obj.get_feature_names_out())
    except Exception:
        return None

def name_accepts(obj, cols):
    return obj is not None and cols is not None and in_cols(obj) == list(cols)

def accepts_any(obj, cols):
    if obj is None or cols is None:
        return False
    ic = in_cols(obj)
    if ic is not None:
        return ic == list(cols)
    n = getattr(obj, "n_features_in_", None)
    if n is None and hasattr(obj, "named_steps"):
        n = getattr(list(obj.named_steps.values())[-1], "n_features_in_", None)
    return n is not None and int(n) == len(cols)

def make_full(model):
    if name_accepts(model, features):
        return model
    if accepts_any(pre, features):
        p_out = out_cols(pre)
        if accepts_any(sel, p_out):
            s_out = out_cols(sel)
            if accepts_any(model, s_out):
                return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])
    if accepts_any(sel, features):
        s_out = out_cols(sel)
        if accepts_any(model, s_out):
            return Pipeline([("selector", sel), ("model", model)])
    if accepts_any(pre, features):
        p_out = out_cols(pre)
        if accepts_any(model, p_out):
            return Pipeline([("preprocessor", pre), ("model", model)])
    raise ValueError(f"Cannot combine saved transforms for {type(model)}; feature_names_in_={in_cols(model)}")

def macro_ovr(y_true, proba, classes):
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    present = [c for c in classes if np.any(y_true == c)]
    missing = [c for c in classes if c not in present]
    if len(present) == len(classes):
        return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    vals = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        vals.append(roc_auc_score(y_true == c, proba[:, j]))
    if missing:
        print(f"missing_classes={len(missing)}")
    return float(np.mean(vals)) if vals else float("nan")

y_col = targets[0]
y_val = val[y_col].to_numpy()
rows, primary_vals = [], {}
for name, model in models:
    full = make_full(model)
    cols = in_cols(full) or features
    Xv = val[cols].copy()
    proba = full.predict_proba(Xv)
    classes = np.asarray(full.classes_)
    pred = full.predict(Xv)
    auc = macro_ovr(y_val, proba, classes)
    f1 = f1_score(y_val, pred, labels=classes, average="macro", zero_division=0)
    rows.append({"model": name, "metric": primary, "value": auc, "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": f1, "threshold": np.nan})
    primary_vals[name] = auc

lower_better = primary.lower() in {"log_loss", "brier_score", "brier", "neg_log_loss", "mean_squared_error", "mae", "rmse"}
best_name = min(primary_vals, key=primary_vals.get) if lower_better else max(primary_vals, key=primary_vals.get)
best_value = float(primary_vals[best_name])
out = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
out["value"] = out["value"].map(lambda x: float(f"{x:.4g}"))
out["threshold"] = np.nan
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics for candidate models")

mlca_runtime.record_metric(primary, best_value, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
if primary in {"roc_auc_ovr", "roc_auc"}:
    mlca_runtime.record_fact_candidate("best_val_roc_auc", best_value, source="validation_metrics.csv")

print(f"features={len(features)} train={len(train)} val={len(val)}")
print(f"best_model={best_name} {primary}={best_value:.4g}")
