import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score


def need(d, key, where):
    if key not in d:
        raise KeyError(f"Missing {where}.{key}; available keys: {sorted(d)}")
    return d[key]


def sig(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan


def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise AttributeError("model has no classes_")


def macro_ovr(y, proba, classes):
    y = np.asarray(y)
    present = [c for c in classes if np.any(y == c)]
    missing = [c for c in classes if c not in present]
    if len(present) == len(classes):
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)), missing
    vals = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        vals.append(roc_auc_score((y == c).astype(int), proba[:, j]))
    return float(np.mean(vals)), missing


mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = need(protocol, "targets", "_protocol")
excluded = need(protocol, "excluded_columns", "_protocol")
group = protocol.get("group_column")
task = need(protocol, "task", "_protocol")
primary = need(task, "primary_metric", "_protocol.task")
need(task, "type", "_protocol.task")

if len(targets) != 1:
    raise ValueError("expected exactly one target column")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

exclude_cols = set(targets) | set(excluded)
if group:
    exclude_cols.add(group)

features = []
for col in train.columns:
    if col in exclude_cols:
        continue
    s = train[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    features.append(col)
if not features:
    raise ValueError("no feature columns remain after exclusions")

model_keys = [
    ("logistic", "07_model_train_logistic_Z7HS8D6N/model.pkl"),
    ("alternative", "08_model_train_alternative_MG1H66HT/alternative_model.pkl"),
]
rows = []
primary_vals = {}

for name, key in model_keys:
    with open(inp[key], "rb") as f:
        model = pickle.load(f)
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        cols = [str(c) for c in model.feature_names_in_]
    else:
        cols = list(features)
    if any(c in exclude_cols for c in cols):
        raise ValueError("target or excluded column present in model features")

    X = val.loc[:, cols]
    y = val[target]
    proba = model.predict_proba(X)
    classes = get_classes(model)
    y_pred = model.predict(X)

    auc, missing = macro_ovr(y, proba, classes)
    f1 = float(f1_score(y, y_pred, average="macro", labels=classes))
    primary_vals[name] = auc

    rows.append({"model": name, "metric": primary, "value": sig(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": sig(f1), "threshold": np.nan})
    mlca_runtime.record_metric(primary, float(auc), split="val")
    if missing:
        print(f"{name}: classes left out of macro OVR: {missing}")

lower_is_better = primary.lower() in {"brier_score", "log_loss", "mae", "mse", "rmse", "mean_squared_error"}
best_name = min(primary_vals, key=primary_vals.get) if lower_is_better else max(primary_vals, key=primary_vals.get)
best_val = primary_vals[best_name]

out = pd.DataFrame(rows)[["model", "metric", "value", "threshold"]]
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics; multiclass threshold not applicable.")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_val), source="validation_metrics.csv")

print(f"train {train.shape} val {val.shape}")
print(f"features {len(features)}")
print(f"best {best_name} {primary}={best_val:.4f}")
