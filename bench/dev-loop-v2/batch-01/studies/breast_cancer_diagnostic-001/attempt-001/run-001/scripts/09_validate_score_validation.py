import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
task = proto["task"]
primary = task["primary_metric"]
target = targets[0]
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
base_excl = set(targets) | set(excluded) | ({group} if group else set())
cand_cols = [c for c in train.columns if c not in base_excl]
key_like = []
for c in cand_cols:
    s = train[c]
    if ((s.dtype == object) or str(s.dtype).startswith("string") or pd.api.types.is_integer_dtype(s)):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
feature_cols = [c for c in cand_cols if c not in key_like]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def classes_of(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return np.asarray(last.classes_)
    raise RuntimeError("Model has no classes_")

def features_of(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return feature_cols

candidates = [
    ("07_model_train_logistic_2M248THN", inp["07_model_train_logistic_2M248THN/model.pkl"]),
    ("08_model_train_alternative_G1J7WVC3", inp["08_model_train_alternative_G1J7WVC3/alternative_model.pkl"]),
]
if inp["latest_model"].resolve() != candidates[1][1].resolve():
    candidates.append(("latest_model", inp["latest_model"]))

y_val = val[target]
results, rows = [], []
for name, path in candidates:
    model = load_pkl(path)
    X = val.loc[:, features_of(model)]
    proba = model.predict_proba(X)
    classes = classes_of(model)
    if len(classes) != 2:
        raise ValueError("Binary validation expected exactly two model classes")
    pos = task.get("positive_label", classes[1])
    if pos not in classes:
        pos = classes[1]
    p = proba[:, int(np.where(classes == pos)[0][0])]
    yb = (y_val == pos).astype(int).to_numpy()
    fpr, tpr, thr = roc_curve(yb, p)
    finite = np.isfinite(thr)
    j = tpr[finite] - fpr[finite]
    threshold = float(thr[finite][int(np.argmax(j))]) if finite.any() else 0.5
    pred = (p >= threshold).astype(int)
    auc = float(roc_auc_score(yb, p))
    pr = float(average_precision_score(yb, p))
    ba = float(balanced_accuracy_score(yb, pred))
    bs = float(brier_score_loss(yb, p))
    results.append({"model": name, "primary": auc, "threshold": threshold})
    for metric, value in [
        (primary, auc), ("pr_auc", pr), ("balanced_accuracy", ba), ("brier_score", bs)
    ]:
        rows.append({
            "model": name,
            "metric": metric,
            "value": float(f"{value:.4g}"),
            "threshold": float(f"{threshold:.4g}"),
        })

mlca_runtime.save_table(
    pd.DataFrame(rows),
    "validation_metrics.csv",
    caption="Validation metrics by candidate model.",
)
for r in sorted(results, key=lambda x: x["primary"]):
    mlca_runtime.record_metric(primary, r["primary"], split="val")
best = max(results, key=lambda x: x["primary"])
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best["primary"], source="validation_metrics.csv")
print(f"val shape {val.shape}, candidates {len(candidates)}")
print(f"best {best['model']} {primary}={best['primary']:.4f} threshold={best['threshold']:.4f}")
