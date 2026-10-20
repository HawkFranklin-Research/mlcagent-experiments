import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary = task["primary_metric"]
if task.get("type") != "multiclass":
    raise ValueError(f"Unsupported task type: {task.get('type')}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y = val[target]


def key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False


drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
feature_cols = [c for c in feature_cols if not key_like(train[c])]


def classes_of(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if isinstance(m, Pipeline) and hasattr(m.steps[-1][1], "classes_"):
        return np.asarray(m.steps[-1][1].classes_)
    raise AttributeError("classes_ not found")


def model_columns(m):
    return list(getattr(m, "feature_names_in_", feature_cols))


def macro_ovr_auc(y_true, proba, classes):
    present = [c for c in classes if np.any(y_true == c)]
    if len(present) == len(classes):
        return roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=classes
        )
    vals = []
    for j, c in enumerate(classes):
        if np.any(y_true == c):
            vals.append(roc_auc_score((y_true == c).astype(int), proba[:, j]))
    missing = [c for c in classes if not np.any(y_true == c)]
    if missing:
        print("missing val classes:", missing)
    return float(np.mean(vals)) if vals else np.nan


model_paths = {
    "logistic": inputs["08_model_train_logistic_9FB709K5/model.pkl"],
    "alternative": inputs["09_model_train_alternative_8G8NH389/alternative_model.pkl"],
}
rows = []
scores = {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    X = val[model_columns(model)]
    proba = model.predict_proba(X)
    classes = classes_of(model)
    auc = macro_ovr_auc(y.to_numpy(), proba, classes)
    pred = model.predict(X) if hasattr(model, "predict") else classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y, pred, average="macro", labels=classes, zero_division=0)
    rows.append({"model": name, "metric": primary, "value": round(float(auc), 4), "threshold": np.nan})
    rows.append({"model": name, "metric": "f1_macro", "value": round(float(f1), 4), "threshold": np.nan})
    scores[name] = auc

metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics")

higher_better = primary not in {"brier"}
best_model = max(scores, key=scores.get) if higher_better else min(scores, key=scores.get)
best_auc = float(scores[best_model])
mlca_runtime.record_metric(primary, best_auc, "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print("train", train.shape, "val", val.shape, "features", len(feature_cols))
print("best", best_model, "val", primary, round(best_auc, 4))
