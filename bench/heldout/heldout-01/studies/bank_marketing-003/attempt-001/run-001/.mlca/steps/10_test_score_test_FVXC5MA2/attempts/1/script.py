import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, roc_curve

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"missing {k}; available top-level keys: {list(proto)}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group_col = proto["group_column"]
task = proto["task"]
task_type = str(task.get("type", "")).lower()
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    is_keylike = s.is_unique and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    )
    if not is_keylike:
        feature_cols.append(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl(inputs["05_preprocess_fit_preprocessing_KGJNVG5D/preprocessor.pkl"])
selector = load_pkl(inputs["06_features_select_features_20NTEHHR/selector.pkl"])

def has_steps(obj, names):
    return isinstance(obj, Pipeline) and any(n in obj.named_steps for n in names)

def make_final(model):
    if isinstance(model, Pipeline) and (
        len(model.named_steps) > 1 or has_steps(model, ["preprocessor", "preprocess", "preprocessing", "selector", "select", "features"])
    ):
        return model
    if has_steps(selector, ["preprocessor", "preprocess", "preprocessing"]):
        return Pipeline([("front", selector), ("model", model)])
    return Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", model)])

def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if isinstance(model, Pipeline):
        last = model.named_steps[list(model.named_steps)[-1]]
        return np.asarray(last.classes_)
    raise AttributeError("model has no classes_")

def input_cols(model):
    cols = list(getattr(model, "feature_names_in_", []))
    return cols if cols else feature_cols

def find_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "threshold" in k.lower():
                try:
                    f = float(v)
                    if 0 <= f <= 1:
                        return f
                except Exception:
                    pass
            r = find_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_threshold(v)
            if r is not None:
                return r
    return None

val = pd.read_parquet(inputs["val"])
val_manifest = json.loads(inputs["09_validate_score_validation_T15DR8CA/manifest.json"].read_text())
target = targets[0]
y_val = val[target].to_numpy()

candidates = [
    ("logistic", inputs["07_model_train_logistic_VP57BH50/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_RH6N3E81/alternative_model.pkl"]),
]
best_name, best_model, best_score = None, None, None
higher_better = primary_metric.lower() not in {"brier_score", "log_loss", "mse", "mae"}

for name, path in candidates:
    model = make_final(load_pkl(path))
    classes = get_classes(model)
    X_val = val[input_cols(model)]
    proba = model.predict_proba(X_val)
    if task_type == "binary":
        pos_label = task.get("positive_label", classes[1] if len(classes) > 1 else classes[0])
        pos_idx = int(np.where(classes == pos_label)[0][0]) if pos_label in classes else 1
        yb = (y_val == pos_label).astype(int)
        if primary_metric in {"roc_auc", "roc_auc_score"}:
            score = roc_auc_score(yb, proba[:, pos_idx])
        elif primary_metric in {"pr_auc", "average_precision"}:
            score = average_precision_score(yb, proba[:, pos_idx])
        elif primary_metric == "balanced_accuracy":
            pred = classes[(proba[:, pos_idx] >= 0.5).astype(int)]
            score = balanced_accuracy_score(y_val, pred)
        else:
            raise ValueError(f"unsupported binary primary_metric: {primary_metric}")
    else:
        if primary_metric in {"roc_auc", "roc_auc_ovr", "macro_roc_auc"}:
            vals = [roc_auc_score((y_val == c).astype(int), proba[:, j]) for j, c in enumerate(classes) if np.any(y_val == c)]
            score = float(np.mean(vals)) if vals else np.nan
        elif primary_metric in {"f1_macro", "f1"}:
            score = f1_score(y_val, classes[np.argmax(proba, axis=1)], average="macro", labels=classes)
        else:
            raise ValueError(f"unsupported multiclass primary_metric: {primary_metric}")
    if best_score is None or (score > best_score if higher_better else score < best_score):
        best_name, best_model, best_score = name, model, score

classes = get_classes(best_model)
X_val = val[input_cols(best_model)]
proba_val = best_model.predict_proba(X_val)
if task_type == "binary":
    pos_label = task.get("positive_label", classes[1] if len(classes) > 1 else classes[0])
    pos_idx = int(np.where(classes == pos_label)[0][0]) if pos_label in classes else 1
    threshold = find_threshold(val_manifest)
    if threshold is None:
        yb_val = (y_val == pos_label).astype(int)
        fpr, tpr, ths = roc_curve(yb_val, proba_val[:, pos_idx])
        finite = np.isfinite(ths)
        threshold = float(ths[finite][np.argmax((tpr - fpr)[finite])])
else:
    threshold, pos_idx, pos_label = None, None, None

test = pd.read_parquet(inputs["test"])
X_test = test[input_cols(best_model)]
proba = best_model.predict_proba(X_test)
y_true = test[target].to_numpy()

def calc_metrics(y, p):
    if task_type == "binary":
        yb = (y == pos_label).astype(int)
        score = p[:, pos_idx]
        pred_idx = np.where(score >= threshold, pos_idx, 1 - pos_idx)
        yp = classes[pred_idx]
        ypb = (yp == pos_label).astype(int)
        res = {}
        try:
            res["roc_auc"] = roc_auc_score(yb, score) if len(np.unique(yb)) == 2 else np.nan
        except Exception:
            res["roc_auc"] = np.nan
        try:
            res["pr_auc"] = average_precision_score(yb, score) if len(np.unique(yb)) == 2 else np.nan
        except Exception:
            res["pr_auc"] = np.nan
        try:
            res["balanced_accuracy"] = balanced_accuracy_score(y, yp)
        except Exception:
            res["balanced_accuracy"] = np.nan
        res["brier_score"] = float(np.mean((score - yb) ** 2))
        tp = np.sum((yb == 1) & (ypb == 1)); fn = np.sum((yb == 1) & (ypb == 0))
        tn = np.sum((yb == 0) & (ypb == 0)); fp = np.sum((yb == 0) & (ypb == 1))
        res["sensitivity"] = tp / (tp + fn) if tp + fn > 0 else np.nan
        res["specificity"] = tn / (tn + fp) if tn + fp > 0 else np.nan
        return res
    yp = classes[np.argmax(p, axis=1)]
    vals = []
    for j, c in enumerate(classes):
        yb = (y == c).astype(int)
        if np.any(y == c) and len(np.unique(yb)) == 2:
            vals.append(roc_auc_score(yb, p[:, j]))
    res = {"roc_auc_ovr": float(np.mean(vals)) if vals else np.nan}
    try:
        res["f1_macro"] = f1_score(y, yp, average="macro", labels=classes)
    except Exception:
        res["f1_macro"] = np.nan
    return res

full = calc_metrics(y_true, proba)
rng = np.random.default_rng(42)
boot = {k: [] for k in full}
n = len(y_true)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    r = calc_metrics(y_true[idx], proba[idx])
    for k in boot:
        boot[k].append(r.get(k, np.nan))

ci = {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in boot.items()}
norm = {
    "roc_auc": "roc_auc", "pr_auc": "pr_auc", "average_precision": "pr_auc",
    "balanced_accuracy": "balanced_accuracy", "brier_score": "brier_score",
    "sensitivity": "sensitivity", "specificity": "specificity",
    "roc_auc_ovr": "roc_auc_ovr", "macro_roc_auc": "roc_auc_ovr",
    "f1_macro": "f1_macro", "f1": "f1_macro",
}
primary_key = norm.get(primary_metric, primary_metric if primary_metric in full else None)
if primary_key is None or primary_key not in full:
    raise ValueError(f"declared primary_metric not computed: {primary_metric}")

rows, seen = [], set()
rows.append((primary_metric, full[primary_key], ci[primary_key][0], ci[primary_key][1]))
seen.add(primary_key)
for k, v in full.items():
    if k not in seen:
        rows.append((k, v, ci[k][0], ci[k][1]))

def r4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

metrics_df = pd.DataFrame([(m, r4(v), r4(lo), r4(hi)) for m, v, lo, hi in rows],
                          columns=["metric", "value", "ci_low", "ci_high"])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Frozen model test metrics with 95% bootstrap CI")

out = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y_true})
if task_type == "binary":
    pred_idx = np.where(proba[:, pos_idx] >= threshold, pos_idx, 1 - pos_idx)
    out["y_pred"] = classes[pred_idx]
else:
    out["y_pred"] = classes[np.argmax(proba, axis=1)]
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

primary_val = float(full[primary_key])
primary_lo, primary_hi = float(ci[primary_key][0]), float(ci[primary_key][1])
mlca_runtime.record_metric(primary_metric, primary_val, "test", ci=(primary_lo, primary_hi))
if primary_metric == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", primary_val, source="test_metrics.csv")

print(f"selected={best_name} n_test={len(test)} threshold={threshold}")
print(f"{primary_metric}={primary_val:.4f} ci=({primary_lo:.4f},{primary_hi:.4f})")
