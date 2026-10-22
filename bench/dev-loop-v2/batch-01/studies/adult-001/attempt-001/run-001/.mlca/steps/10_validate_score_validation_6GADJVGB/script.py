import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
primary = task["primary_metric"]
pos_label = task.get("positive_label", protocol.get("positive_label", None))

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y_val = val[targets[0]].values

def key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_integer_dtype(s))

skip = set(targets + excluded + ([group] if group else []))
feature_cols = [c for c in train.columns if c not in skip and not key_like(train[c])]
Xv = val[feature_cols].copy()

pre = pickle.loads(inp["06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["07_features_select_features_Y03WXMRF/selector.pkl"].read_bytes())
models = {
    "logistic": pickle.loads(inp["08_model_train_logistic_MMDVH86M/model.pkl"].read_bytes()),
    "alternative": pickle.loads(inp["09_model_train_alternative_4GY8FEXD/alternative_model.pkl"].read_bytes()),
}

def classes_of(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline):
        return np.asarray(obj.named_steps[obj.steps[-1][0]].classes_)
    raise AttributeError("no classes_")

def proba_of(obj, X):
    raw = [c for c in feature_cols if c in X.columns]
    try:
        return obj.predict_proba(X[raw])
    except Exception:
        pass
    if hasattr(obj, "feature_names_in_"):
        exp = [c for c in obj.feature_names_in_ if c in X.columns]
        if len(exp) == len(obj.feature_names_in_):
            try:
                return obj.predict_proba(X[exp])
            except Exception:
                pass
    if hasattr(sel, "feature_names_in_"):
        sin = [c for c in sel.feature_names_in_ if c in X.columns]
        if len(sin) == len(sel.feature_names_in_):
            try:
                return Pipeline([("sel", sel), ("model", obj)]).predict_proba(X[sin])
            except Exception:
                pass
    pin = [c for c in getattr(pre, "feature_names_in_", raw) if c in feature_cols]
    pipe = Pipeline([("pre", pre), ("sel", sel), ("model", obj)])
    return pipe.predict_proba(X[pin])

def binary_metrics(y, p, classes):
    pl = pos_label if pos_label in classes else classes[1]
    j = int(np.where(classes == pl)[0][0])
    pp = p[:, j]
    yi = (y == pl).astype(int)
    auc = roc_auc_score(yi, pp)
    ap = average_precision_score(yi, pp)
    brier = brier_score_loss(yi, pp)
    fpr, tpr, thr = roc_curve(yi, pp)
    t = float(thr[int(np.argmax(tpr - fpr))])
    pred = (pp >= t).astype(int)
    bal = balanced_accuracy_score(yi, pred)
    return {"roc_auc": auc, "pr_auc": ap, "balanced_accuracy": bal, "brier_score": brier}, t

def multi_metrics(y, p, classes):
    pred = classes[np.argmax(p, axis=1)]
    f1 = f1_score(y, pred, average="macro", labels=classes)
    present = [c for c in classes if c in set(y)]
    vals = [roc_auc_score((y == c).astype(int), p[:, int(np.where(classes == c)[0][0])]) for c in present]
    return {"roc_auc": float(np.mean(vals)), "f1_macro": f1}, np.nan

rows, aucs = [], {}
for name, obj in models.items():
    classes = classes_of(obj)
    p = proba_of(obj, Xv)
    if len(classes) > 2:
        met, thr = multi_metrics(y_val, p, classes)
    else:
        met, thr = binary_metrics(y_val, p, classes)
    for k, v in met.items():
        rows.append({"model": name, "metric": k, "value": float(f"{v:.4g}"),
                     "threshold": thr if not np.isnan(thr) else np.nan})
    aucs[name] = met["roc_auc"]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics")

lower = primary in {"brier_score", "log_loss", "rmse", "mae", "mse"}
primary_vals = {r["model"]: r["value"] for r in rows if r["metric"] == primary}
best_model = min(primary_vals, key=primary_vals.get) if lower else max(primary_vals, key=primary_vals.get)
best_primary = primary_vals[best_model]
best_auc = aucs[best_model]

mlca_runtime.record_metric(primary, best_primary, split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best_model={best_model} {primary}={best_primary:.4f} roc_auc={best_auc:.4f}")
