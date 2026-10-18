import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
try:
    targets = proto["targets"]; excluded = proto["excluded_columns"]
    task = proto["task"]; pm = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing {e}; keys={list(proto)}")
if not pm:
    raise ValueError("primary_metric missing or null")
group = proto.get("group_column")

train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"])
skip = set(targets) | set(excluded)
if group: skip.add(group)
feature_cols = [c for c in train.columns if c not in skip and not (
    train[c].nunique(dropna=False) == len(train) and (
        train[c].dtype == object or pd.api.types.is_string_dtype(train[c])
        or pd.api.types.is_integer_dtype(train[c])))]
target_col = targets[0]
y_val = val[target_col]

def load(p):
    with open(p, "rb") as f: return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_02TSADV2/preprocessor.pkl"])
sel = load(inp["06_features_select_features_BS9MRTV1/selector.pkl"])
models = [("logistic", load(inp["07_model_train_logistic_8ET5BK8M/model.pkl"])),
          ("alternative", load(inp["08_model_train_alternative_TE7YJ5NR/alternative_model.pkl"]))]

def final_est(o):
    return list(o.named_steps.values())[-1] if isinstance(o, Pipeline) else o

def classes_of(o):
    cl = getattr(o, "classes_", None)
    return np.asarray(cl if cl is not None else final_est(o).classes_)

def selector_is_frontend(selector, preprocessor):
    if isinstance(selector, Pipeline):
        if any(v is preprocessor for v in selector.named_steps.values()): return True
        names = [n.lower() for n in selector.named_steps]
        if any(("preprocess" in n) or ("imputer" in n) or ("encoder" in n) or ("column" in n) for n in names):
            return True
    return False

def build_predictor(model, preprocessor, selector, feature_cols):
    if isinstance(model, Pipeline):
        names = [n.lower() for n in model.named_steps]
        if any(("preprocess" in n) or ("select" in n) or ("imputer" in n) or ("encoder" in n) or ("column" in n) for n in names):
            return model
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None and set(feature_cols).issubset(set(fn)): return model
    front = selector if selector_is_frontend(selector, preprocessor) else Pipeline(
        [("preprocessor", preprocessor), ("selector", selector)])
    return Pipeline([("frontend", front), ("model", model)])

is_binary = task.get("type") == "binary" or task.get("n_classes") == 2
rows, primary = [], []
pos_used = None
for name, model in models:
    predictor = build_predictor(model, pre, sel, feature_cols)
    fn = getattr(predictor, "feature_names_in_", None)
    cols = list(fn) if fn is not None else feature_cols
    X = val[cols].copy()
    proba = predictor.predict_proba(X)
    cl = classes_of(predictor)
    if is_binary:
        pos = task.get("positive_label") or cl[1]
        pos_used = pos
        prob_pos = proba[:, int(np.where(cl == pos)[0][0])]
        y_bin = (y_val == pos).astype(int).to_numpy()
        fpr, tpr, thr = roc_curve(y_bin, prob_pos)
        finite = np.isfinite(thr); j = tpr - fpr
        idx = np.where(finite)[0][np.argmax(j[finite])] if finite.any() else 0
        theta = float(thr[idx]) if finite.any() else 0.5
        pred = (prob_pos >= theta).astype(int)
        metrics = {"roc_auc": roc_auc_score(y_bin, prob_pos),
                   "pr_auc": average_precision_score(y_bin, prob_pos),
                   "balanced_accuracy": balanced_accuracy_score(y_bin, pred),
                   "brier_score": brier_score_loss(y_bin, prob_pos)}
        for m, v in metrics.items():
            rows.append({"model": name, "metric": m, "value": float(v),
                         "threshold": theta if m == "balanced_accuracy" else np.nan})
    else:
        y = y_val.to_numpy()
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=cl)
        pred = cl[np.argmax(proba, axis=1)]
        metrics = {"roc_auc_ovr": auc,
                   "f1_macro": f1_score(y, pred, average="macro", labels=cl)}
        for m, v in metrics.items():
            rows.append({"model": name, "metric": m, "value": float(v), "threshold": np.nan})
    primary.append((metrics.get(pm), name, metrics))
    mlca_runtime.record_metric(pm, float(metrics[pm]), "val")

minimize = any(k in pm.lower() for k in ["brier", "loss", "error"])
valid = [(v, n, m) for v, n, m in primary if v is not None and not pd.isna(v)]
if not valid: raise ValueError("No validation values for declared primary metric")
best_v, best_n, best_m = (min(valid, key=lambda t: t[0]) if minimize
                          else max(valid, key=lambda t: t[0]))

out = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
out["value"] = out["value"].astype(float).round(4)
out["threshold"] = out["threshold"].astype(float)
cap = f"Validation metrics; positive={pos_used}" if is_binary else "Validation metrics"
mlca_runtime.save_table(out, "validation_metrics.csv", caption=cap)
mlca_runtime.record_fact_candidate("best_model", best_n, source="validation_metrics.csv")
best_auc = best_m.get("roc_auc", best_m.get("roc_auc_ovr", best_v))
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")
print(len(feature_cols), len(val), best_n, round(float(best_v), 4))
print("saved validation_metrics.csv")
