import json, pickle
import numpy as np, pandas as pd
import mlca_runtime as rt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, auc as auc_fn,
                             f1_score, accuracy_score, balanced_accuracy_score, brier_score_loss)

rt.set_seed(42)
inp = rt.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]; pm = task["primary_metric"]; ttype = str(task.get("type", "")).lower()
targets = list(proto["targets"]); excluded = list(proto.get("excluded_columns") or [])
group = proto.get("group_column")

train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"])

def keylike(s):
    if s.dtype == object or str(s.dtype).startswith("string"):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        return bool(s.max() - s.min() + 1 == len(s)) or str(s.name).lower() in ("id", "index", "row_id")
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in drop and not keylike(train[c])]
target = targets[0]
y = val[target].to_numpy()

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["06_preprocess_fit_preprocessing_XZ1WZNJH/preprocessor.pkl"])
sel = load(inp["07_features_select_features_KRQ7615Z/selector.pkl"])
models = {"logistic": load(inp["08_model_train_logistic_6YJ0NC7Y/model.pkl"]),
          "alternative": load(inp["09_model_train_alternative_143BSS6M/alternative_model.pkl"])}

def proba(obj, df):
    if hasattr(obj, "predict_proba") and hasattr(obj, "feature_names_in_"):
        return np.asarray(obj.predict_proba(df[list(obj.feature_names_in_)])), np.asarray(obj.classes_)
    pre_cols = list(getattr(pre, "feature_names_in_", features))
    if sel is not None and hasattr(sel, "feature_names_in_") and list(sel.feature_names_in_) == pre_cols:
        X = sel.transform(df[pre_cols])
    else:
        X = pre.transform(df[pre_cols])
        if sel is not None:
            X = sel.transform(X)
    return np.asarray(obj.predict_proba(X)), np.asarray(obj.classes_)

rows, scores, left_out = [], {}, {}
for name, mdl in models.items():
    P, cls = proba(mdl, val)
    if ttype == "binary":
        pos = task.get("positive_label", cls[1]); j = int(np.where(cls == pos)[0][0])
        p = P[:, j]; yb = (y == pos).astype(int)
        prec, rec, _ = precision_recall_curve(yb, p)
        fpr, tpr, thr_grid = roc_curve(yb, p)
        thr = float(thr_grid[int(np.argmax(tpr - fpr))])
        m = {"roc_auc": float(roc_auc_score(yb, p)), "pr_auc": float(auc_fn(rec, prec)),
             "brier": float(brier_score_loss(yb, p)),
             "balanced_accuracy": float(balanced_accuracy_score(yb, (p >= thr).astype(int)))}
        thr_col = {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "balanced_accuracy": thr}
    else:
        present = [c for c in cls if (y == c).any()]
        idx = {c: i for i, c in enumerate(cls)}
        left_out[name] = [c for c in cls if c not in present]
        aucs = [roc_auc_score((y == c).astype(int), P[:, idx[c]]) for c in present]
        pred = cls[np.argmax(P, axis=1)]
        m = {"roc_auc_ovr": float(np.mean(aucs)),
             "f1_macro": float(f1_score(y, pred, average="macro", labels=present)),
             "accuracy": float(accuracy_score(y, pred))}
        thr_col = {k: np.nan for k in m}
    if pm not in m:
        raise KeyError(f"declared primary_metric {pm!r} not computed; available: {sorted(m)}")
    scores[name] = m[pm]
    rows += [{"model": name, "metric": k, "value": round(float(v), 6), "threshold": thr_col[k]}
             for k, v in m.items()]

sign = -1.0 if "brier" in pm else 1.0
best = (max if sign > 0 else min)(scores, key=lambda k: sign * scores[k])
table = pd.DataFrame(rows)[["model", "metric", "value", "threshold"]]
rt.save_table(table, "validation_metrics.csv", "Validation metrics per candidate model (multiclass: no binary threshold)")
for name, v in scores.items():
    rt.record_metric(f"{pm}_{name}", float(v), "val")
rt.record_metric(pm, float(scores[best]), "val")
rt.record_fact_candidate("best_model", best, source="validation_metrics.csv")
rt.record_fact_candidate("best_val_roc_auc", float(scores[best]), source="validation_metrics.csv")

print(f"features={len(features)} val_rows={len(val)} primary_metric={pm}")
print(f"scores={ {k: round(v, 4) for k, v in scores.items()} } best={best} ({sign>0 and 'max' or 'min'})")
print(f"classes_left_out={left_out}")
