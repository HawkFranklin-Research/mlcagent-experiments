import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    balanced_accuracy_score,
    f1_score,
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl")
sel = load_pkl("06_features_select_features_CZZQV6ME/selector.pkl")

def as_pipeline(cand):
    if isinstance(cand, Pipeline):
        if len(cand.named_steps) == 1:
            cand = list(cand.named_steps.values())[0]
        else:
            return cand
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", cand))
    return Pipeline(steps)

models = {
    "logistic": as_pipeline(load_pkl("07_model_train_logistic_J5YKBG0Y/model.pkl")),
    "alternative": as_pipeline(load_pkl("08_model_train_alternative_P5WQW45K/alternative_model.pkl")),
}

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

expected = None
for m in models.values():
    expected = input_names(m)
    if expected:
        break

if expected:
    feature_cols = list(expected)
else:
    protected = set(targets) | set(excluded)
    if group:
        protected.add(group)

    def keylike(s):
        if not s.notna().all() or not s.is_unique:
            return False
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return True
        if pd.api.types.is_integer_dtype(s):
            d = s.diff().dropna()
            return len(d) > 0 and (d == 1).all()
        return False

    feature_cols = [
        c for c in train.columns
        if c not in protected and not keylike(train[c])
    ]

missing = [c for c in feature_cols if c not in train.columns or c not in val.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X_val = val[feature_cols]
y_val = val[targets[0]]
n_classes = int(task.get("n_classes", 2))

def classes_of(m):
    if hasattr(m, "classes_") and m.classes_ is not None:
        return np.asarray(m.classes_)
    return np.asarray(m.named_steps[list(m.named_steps)[-1]].classes_)

rows, results = [], {}
for name, m in models.items():
    proba = np.asarray(m.predict_proba(X_val))
    cls = classes_of(m)

    if n_classes == 2:
        pos = task.get("positive_label", cls[1])
        if pos not in cls:
            pos = cls[1]
        p = proba[:, int(np.where(cls == pos)[0][0])]
        yb = (y_val == pos).astype(int).to_numpy()

        roc = float(roc_auc_score(yb, p))
        pr = float(average_precision_score(yb, p))
        brier = float(brier_score_loss(yb, p))

        fpr, tpr, thr = roc_curve(yb, p)
        j = tpr - fpr
        mask = np.isfinite(thr)
        youden = float(thr[mask][np.argmax(j[mask])]) if mask.any() else 0.5

        pred = (p >= youden).astype(int)
        bal = float(balanced_accuracy_score(yb, pred))

        results[name] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "balanced_accuracy": bal,
            "brier_score": brier,
        }
        for met, v in [
            ("roc_auc", roc),
            ("pr_auc", pr),
            ("balanced_accuracy", bal),
            ("brier_score", brier),
        ]:
            rows.append({"model": name, "metric": met, "value": v, "threshold": youden})
    else:
        pred = m.predict(X_val)
        roc = float(roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=cls))
        f1 = float(f1_score(y_val, pred, average="macro"))
        results[name] = {"roc_auc": roc, "f1_macro": f1}
        for met, v in [("roc_auc_ovr_macro", roc), ("f1_macro", f1)]:
            rows.append({"model": name, "metric": met, "value": v, "threshold": np.nan})

higher_is_better = primary not in {"brier_score", "log_loss", "rmse", "mae"}
best_name = (
    max(results, key=lambda n: results[n][primary])
    if higher_is_better
    else min(results, key=lambda n: results[n][primary])
)
best_val = float(results[best_name][primary])
best_auc = float(results[best_name]["roc_auc"])

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
for col in ["value", "threshold"]:
    df[col] = df[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model")

for r in rows:
    mlca_runtime.record_metric(f"{r['metric']}_{r['model']}", float(r["value"]), split="val")
mlca_runtime.record_metric(primary, best_val, split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train {train.shape} val {val.shape}")
print(f"features {len(feature_cols)}")
print(f"best {best_name} {primary}={best_val:.4g} roc_auc={best_auc:.4g}")
print("saved validation_metrics.csv")
