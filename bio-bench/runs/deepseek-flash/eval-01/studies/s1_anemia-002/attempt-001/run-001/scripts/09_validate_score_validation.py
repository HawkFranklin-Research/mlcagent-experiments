import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise ValueError(f"Missing protocol key {k}; available: {list(proto.keys())}")
targets = list(proto["targets"])
target = targets[0]
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
task = proto["task"]
primary = task["primary_metric"]
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s, name):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return any(tok in name.lower() for tok in ("id", "code", "key", "seq", "index"))
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group
                and not is_keylike(train[c], str(c))]

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

model1 = load_pkl("07_model_train_logistic_4DKW1RVE/model.pkl")
model2 = load_pkl("08_model_train_alternative_7HSX9CPP/alternative_model.pkl")
candidates = {"logistic": model1, "alternative": model2}

def get_classes(est):
    if hasattr(est, "classes_"):
        return list(np.asarray(est.classes_))
    if hasattr(est, "named_steps"):
        for step in reversed(list(est.named_steps.values())):
            if hasattr(step, "classes_"):
                return list(np.asarray(step.classes_))
    return None

def predictor_for(model):
    if hasattr(model, "predict_proba"):
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None:
            cols = list(cols)
            if all(c in val.columns for c in cols):
                return model, cols
        if not hasattr(model, "named_steps"):
            return model, feature_cols
    pre = load_pkl("05_preprocess_fit_preprocessing_YDX4CF7Z/preprocessor.pkl")
    sel = load_pkl("06_features_select_features_FT85WVQ5/selector.pkl")
    sel_cols = getattr(sel, "feature_names_in_", None)
    if sel_cols is not None and all(c in val.columns for c in list(sel_cols)):
        return Pipeline([("selector", sel), ("model", model)]), list(sel_cols)
    pre_cols = getattr(pre, "feature_names_in_", None)
    if pre_cols is not None and all(c in val.columns for c in list(pre_cols)):
        return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)]), list(pre_cols)
    return model, feature_cols

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

rows, results = [], {}
y = val[target].to_numpy()

for name, model in candidates.items():
    pred, cols = predictor_for(model)
    if any(c in targets or c in excluded for c in cols):
        raise ValueError("Model input columns include target or excluded columns")
    X = val[cols].copy()
    proba = pred.predict_proba(X)
    classes = get_classes(pred) or get_classes(model)
    if classes is None:
        raise ValueError("Cannot determine model classes")
    if n_classes == 2:
        pos = task.get("positive_label")
        if pos is None:
            pos = classes[1]
        pos_idx = classes.index(pos)
        ybin = (y == pos).astype(int)
        p = proba[:, pos_idx]
        roc = roc_auc_score(ybin, p)
        pr = average_precision_score(ybin, p)
        fpr, tpr, thr = roc_curve(ybin, p)
        j = tpr - fpr
        finite = np.isfinite(thr)
        if finite.any():
            th = float(thr[finite][int(np.argmax(j[finite]))])
        else:
            th = float(thr[int(np.argmax(j))])
        yhat = (p >= th).astype(int)
        ba = balanced_accuracy_score(ybin, yhat)
        brier = brier_score_loss(ybin, p)
        vals = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": ba, "brier_score": brier}
        if primary not in vals:
            vals[primary] = roc
        for metric, value in vals.items():
            rows.append({"model": name, "metric": metric, "value": sig4(value),
                         "threshold": sig4(th) if metric == "balanced_accuracy" else np.nan})
        results[name] = vals
    else:
        y_pred = pred.predict(X)
        f1 = f1_score(y, y_pred, labels=classes, average="macro", zero_division=0)
        present = [c for c in classes if np.any(y == c)]
        if len(present) == len(classes):
            roc = roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro")
        else:
            aucs = []
            for c in present:
                aucs.append(roc_auc_score((y == c).astype(int), proba[:, classes.index(c)]))
            roc = float(np.mean(aucs))
            print(f"classes missing from val: {[c for c in classes if c not in present]}")
        vals = {"roc_auc": roc, "f1_macro": f1}
        if primary not in vals:
            vals[primary] = roc
        for metric, value in vals.items():
            rows.append({"model": name, "metric": metric, "value": sig4(value), "threshold": np.nan})
        results[name] = vals

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation metrics by candidate model")

higher = ("auc" in primary) or primary in ("f1", "f1_macro", "balanced_accuracy", "accuracy")
best_name = (max if higher else min)(results, key=lambda n: results[n][primary])
best_roc = results[best_name]["roc_auc"]
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", sig4(best_roc), source="validation_metrics.csv")
for name, vals in results.items():
    mlca_runtime.record_metric(f"{primary}:{name}", float(vals[primary]), split="val")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"candidates={list(candidates)}")
print(f"best={best_name} {primary}={results[best_name][primary]:.4f} roc_auc={best_roc:.4f}")
