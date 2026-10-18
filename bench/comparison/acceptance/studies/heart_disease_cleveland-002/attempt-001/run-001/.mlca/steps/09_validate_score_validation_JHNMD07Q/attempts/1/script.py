import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
try:
    targets = proto["targets"]
    excluded = proto["excluded_columns"]
    task = proto["task"]
    pm = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto)}")
if not pm:
    raise ValueError("primary_metric is missing or null")
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
skip = set(targets) | set(excluded)
if group:
    skip.add(group)
feature_cols = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
        continue
    feature_cols.append(c)
target_col = targets[0]
y_val = val[target_col]
X_val = val[feature_cols].copy()

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_02TSADV2/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_BS9MRTV1/selector.pkl"])
models = [
    ("logistic", load_pkl(inp["07_model_train_logistic_8ET5BK8M/model.pkl"])),
    ("alternative", load_pkl(inp["08_model_train_alternative_TE7YJ5NR/alternative_model.pkl"])),
]

def final_est(obj):
    return list(obj.named_steps.values())[-1] if isinstance(obj, Pipeline) else obj

def selector_is_frontend(selector, preprocessor):
    if isinstance(selector, Pipeline):
        if any(v is preprocessor for v in selector.named_steps.values()):
            return True
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
    if fn is not None and set(feature_cols).issubset(set(fn)):
        return model
    front = selector if selector_is_frontend(selector, preprocessor) else Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    return Pipeline([("frontend", front), ("model", model)])

def classes_of(obj):
    cl = getattr(obj, "classes_", None)
    return np.asarray(cl if cl is not None else final_est(obj).classes_)

n_classes = task.get("n_classes")
is_binary = task.get("type", "") == "binary" or n_classes == 2
rows, primary_vals = [], []
pos_used = None

for name, model in models:
    predictor = build_predictor(model, pre, sel, feature_cols)
    expected = getattr(predictor, "feature_names_in_", None)
    cols = list(expected) if expected is not None and set(expected) == set(feature_cols) else feature_cols
    X = val[cols].copy()
    proba = predictor.predict_proba(X)
    cl = classes_of(predictor)
    if is_binary:
        pos = task.get("positive_label", task.get("positive_class", None))
        if pos is None:
            pos = cl[1]
        pos_used = pos
        pos_idx = int(np.where(cl == pos)[0][0])
        prob_pos = proba[:, pos_idx]
        y_bin = (y_val == pos).astype(int).to_numpy()
        fpr, tpr, thr = roc_curve(y_bin, prob_pos)
        j = tpr - fpr
        finite = np.isfinite(thr)
        idx = np.where(finite)[0][np.argmax(j[finite])] if finite.any() else 0
        threshold = float(thr[idx]) if finite.any() else 0.5
        pred = (prob_pos >= threshold).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_bin, prob_pos),
            "pr_auc": average_precision_score(y_bin, prob_pos),
            "balanced_accuracy": balanced_accuracy_score(y_bin, pred),
            "brier_score": brier_score_loss(y_bin, prob_pos),
        }
        for metric, value in metrics.items():
            rows.append({"model": name, "metric": metric, "value": float(value), "threshold": threshold if metric == "balanced_accuracy" else np.nan})
    else:
        y = y_val.to_numpy()
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=cl)
        pred = cl[np.argmax(proba, axis=1)]
        metrics = {"roc_auc_ovr": auc, "f1_macro": f1_score(y, pred, average="macro", labels=cl)}
        for metric, value in metrics.items():
            rows.append({"model": name, "metric": metric, "value": float(value), "threshold": np.nan})
    primary_vals.append((metrics.get(pm, np.nan), name, metrics))
    mlca_runtime.record_metric(pm, float(metrics.get(pm, np.nan)), f"val:{name}")

minimize = any(k in pm.lower() for k in ["brier", "loss", "error"])
valid = [(v, n, m) for v, n, m in primary_vals if not pd.isna(v)]
if not valid:
    raise ValueError("No validation values for declared primary metric")
best_val, best_name, best_metrics = min(valid, key=lambda t: t[0]) if minimize else max(valid, key=lambda t: t[0])

out = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
out["value"] = out["value"].astype(float).round(4)
out["threshold"] = out["threshold"].astype(float).round(4)
cap = f"Validation metrics; positive label={pos_used}" if is_binary else "Validation metrics"
mlca_runtime.save_table(out, "validation_metrics.csv", caption=cap)
mlca_runtime.record_metric(pm, float(best_val), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
best_auc = best_metrics.get("roc_auc", best_metrics.get("roc_auc_ovr", best_val))
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")
print(len(feature_cols), len(val), best_name, round(float(best_val), 4))
print("saved validation_metrics.csv")
