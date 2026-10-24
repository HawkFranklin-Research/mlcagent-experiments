import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, f1_score, roc_curve)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())
try:
    task = protocol["task"]; primary = task["primary_metric"]
    targets = protocol["targets"]; excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}")
group = protocol.get("group_column")
train = pd.read_parquet(ins["train"]); val = pd.read_parquet(ins["val"])
target = targets[0]
excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excl and not (
    pd.api.types.is_object_dtype(train[c]) and train[c].nunique(dropna=True) == len(train))]

def load(p):
    with open(p, "rb") as f: return pickle.load(f)

cand_paths = {
    "logistic": ins["08_model_train_logistic_487M7W52/model.pkl"],
    "alternative": ins["09_model_train_alternative_HZ1W3A7P/alternative_model.pkl"],
}
models = {k: load(v) for k, v in cand_paths.items()}

def in_cols(m):
    if hasattr(m, "feature_names_in_"): return list(m.feature_names_in_)
    if isinstance(m, Pipeline) and m.steps and hasattr(m.steps[0][1], "feature_names_in_"):
        return list(m.steps[0][1].feature_names_in_)
    return feature_cols

def proba(m, X):
    if hasattr(m, "predict_proba"): return m.predict_proba(X)
    d = m.decision_function(X)
    if d.ndim == 1:
        p = 1.0 / (1.0 + np.exp(-d)); return np.column_stack([1-p, p])
    e = np.exp(d - d.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

def binary_metrics(m, X, y):
    p = proba(m, X); classes = np.asarray(m.classes_)
    pos = task.get("positive_label")
    if pos is None: pos = classes[1]
    j = int(np.where(classes == pos)[0][0]); ppos = p[:, j]
    ybin = (y == pos).astype(int)
    fpr, tpr, thr = roc_curve(ybin, ppos)
    jj = tpr - fpr
    finite = np.isfinite(thr)
    idx = np.where(finite)[0][int(np.argmax(jj[finite]))] if finite.any() else 0
    chosen = float(thr[idx])
    ypred = (ppos >= chosen).astype(int)
    vals = {
        "roc_auc": roc_auc_score(ybin, ppos),
        "pr_auc": average_precision_score(ybin, ppos),
        "balanced_accuracy": balanced_accuracy_score(ybin, ypred),
        "brier_score": brier_score_loss(ybin, ppos),
    }
    return vals, chosen

def multiclass_metrics(m, X, y):
    p = proba(m, X); classes = np.asarray(m.classes_)
    yset = set(y)
    present = [c for c in classes if c in yset]
    if len(present) == len(classes):
        auc = roc_auc_score(y, p, labels=classes, multi_class="ovr", average="macro")
    else:
        vals = [roc_auc_score((y == c).astype(int), p[:, j])
                for j, c in enumerate(classes) if c in yset]
        auc = float(np.mean(vals)) if vals else np.nan
        print("AUC omitted classes:", [c for c in classes if c not in yset])
    ypred = m.predict(X) if hasattr(m, "predict") else classes[p.argmax(1)]
    f1 = f1_score(y, ypred, average="macro", labels=classes, zero_division=0)
    return {"roc_auc": auc, "macro_f1": f1}, np.nan

def r4(v):
    return float(f"{v:.4g}") if np.isfinite(v) else np.nan

rows, res = [], []
for name, m in models.items():
    cols = in_cols(m)
    bad = set(cols) & excl
    if bad: raise ValueError(f"Model {name} expects excluded columns: {sorted(bad)}")
    X = val[cols]; y = val[target]
    if task.get("type") == "binary":
        vals, thr = binary_metrics(m, X, y)
        rows.extend({"model": name, "metric": k, "value": r4(v),
                     "threshold": r4(thr) if k == "balanced_accuracy" else np.nan}
                    for k, v in vals.items())
    elif task.get("type") == "multiclass":
        vals, thr = multiclass_metrics(m, X, y)
        rows.extend({"model": name, "metric": k, "value": r4(v), "threshold": np.nan}
                    for k, v in vals.items())
    else:
        raise ValueError(f"Unsupported task type: {task.get('type')}")
    if primary not in vals:
        raise KeyError(f"Primary metric {primary!r} not computed; available: {list(vals)}")
    res.append({"name": name, "primary_value": vals[primary], "roc_auc": vals["roc_auc"]})

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(df, "validation_metrics.csv",
                        caption="Validation metrics by candidate model.")
lower = {"brier_score", "log_loss", "mse", "mae"}
if primary in lower:
    best_i = min(range(len(res)), key=lambda i: res[i]["primary_value"])
else:
    best_i = max(range(len(res)), key=lambda i: res[i]["primary_value"])
best = res[best_i]
order = [i for i in range(len(res)) if i != best_i] + [best_i]
for i in order:
    mlca_runtime.record_metric(primary, float(res[i]["primary_value"]), split="val")
mlca_runtime.record_fact_candidate("best_model", best["name"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["roc_auc"]),
                                   source="validation_metrics.csv")
print(f"rows train/val: {len(train)}/{len(val)}; features: {len(feature_cols)}")
print(f"best_model={best['name']}; {primary}={best['primary_value']:.4g}")
print(f"best_val_roc_auc={best['roc_auc']:.4g}")
