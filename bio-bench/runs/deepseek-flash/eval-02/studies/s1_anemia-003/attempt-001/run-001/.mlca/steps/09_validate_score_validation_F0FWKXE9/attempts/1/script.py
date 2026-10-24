import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(protocol)}")
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"] if isinstance(protocol["targets"], list) else [protocol["targets"]]
target_col = targets[0]
drop = set(targets) | set(protocol.get("excluded_columns") or [])
if protocol.get("group_column"):
    drop.add(protocol["group_column"])

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
feature_cols = [c for c in train.columns if c not in drop]
for c in list(feature_cols):
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        feature_cols.remove(c)

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_T6A9FCB5/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_WE31FN6Z/selector_all.pkl"])
models = {
    "logistic": load_pkl(inp["07_model_train_logistic_21TNHNVT/model.pkl"]),
    "alternative": load_pkl(inp["08_model_train_alternative_67TQZBPK/alternative_model.pkl"]),
}

def expected_cols(obj):
    if obj is None:
        return None
    fni = getattr(obj, "feature_names_in_", None)
    if fni is None and isinstance(obj, Pipeline):
        for st in obj.named_steps.values():
            fni = getattr(st, "feature_names_in_", None)
            if fni is not None:
                break
    return None if fni is None else list(fni)

def as_frame(x, obj, index, prefix):
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "toarray"):
        x = x.toarray()
    cols = obj.get_feature_names_out() if hasattr(obj, "get_feature_names_out") else [f"{prefix}{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, columns=cols, index=index)

def frontend(X):
    exp_sel = expected_cols(sel)
    if sel is not None and exp_sel is not None and all(c in X.columns for c in exp_sel):
        return as_frame(sel.transform(X[exp_sel]), sel, X.index, "s")
    data = X
    if pre is not None:
        exp_pre = expected_cols(pre)
        data = pre.transform(data[exp_pre] if exp_pre is not None and all(c in data.columns for c in exp_pre) else data[feature_cols])
        data = as_frame(data, pre, X.index, "p")
    if sel is not None:
        exp_sel = expected_cols(sel)
        data = sel.transform(data[exp_sel] if exp_sel is not None and all(c in data.columns for c in exp_sel) else data)
        data = as_frame(data, sel, X.index, "s")
    return data

def predict_proba(obj, X):
    exp = expected_cols(obj)
    if exp is not None and all(c in X.columns for c in exp):
        return np.asarray(obj.predict_proba(X[exp]))
    if exp is None and isinstance(obj, Pipeline):
        return np.asarray(obj.predict_proba(X[feature_cols]))
    data = frontend(X)
    if exp is not None and all(c in data.columns for c in exp):
        data = data[exp]
    return np.asarray(obj.predict_proba(data))

y_val = val[target_col]
results, rows = {}, []
for name, model in models.items():
    proba = predict_proba(model, val)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.sort(train[target_col].unique())
    classes = np.asarray(classes)
    if len(classes) != 2:
        raise ValueError("This validation script expects a binary task.")
    positive_label = task.get("positive_label", task.get("positive_class", classes[1]))
    if positive_label not in classes:
        positive_label = classes[1]
    pos_idx = int(np.where(classes == positive_label)[0][0])
    p = proba[:, pos_idx]
    yb = np.asarray(y_val == positive_label, dtype=int)
    fpr, tpr, thr = roc_curve(yb, p)
    thr_sel = float(thr[np.argmax(tpr - fpr)])
    pred = (p >= thr_sel).astype(int)
    vals = {
        "roc_auc": roc_auc_score(yb, p),
        "pr_auc": average_precision_score(yb, p),
        "balanced_accuracy": balanced_accuracy_score(yb, pred),
        "brier_score": brier_score_loss(yb, p),
    }
    results[name] = {"vals": vals, "threshold": thr_sel}
    seen = set()
    for metric in [primary, "pr_auc", "balanced_accuracy", "brier_score"]:
        if metric in seen:
            continue
        seen.add(metric)
        rows.append({"model": name, "metric": metric, "value": float(vals[metric]),
                     "threshold": thr_sel if metric == "balanced_accuracy" else np.nan})

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
df["value"] = df["value"].map(lambda x: float(f"{x:.4g}"))
df["threshold"] = df["threshold"].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model")

for name, res in results.items():
    mlca_runtime.record_metric(primary, float(res["vals"][primary]), "val")
lower_better = primary in {"brier_score", "log_loss", "mae", "mse", "rmse"}
best_name = min(results, key=lambda n: results[n]["vals"][primary]) if lower_better else max(results, key=lambda n: results[n]["vals"][primary])
best_val_auc = results[best_name]["vals"]["roc_auc"]
mlca_runtime.record_metric(primary, float(results[best_name]["vals"][primary]), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_val_auc), source="validation_metrics.csv")
print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
print(f"best_model={best_name} {primary}={results[best_name]['vals'][primary]:.4f} roc_auc={best_val_auc:.4f}")
