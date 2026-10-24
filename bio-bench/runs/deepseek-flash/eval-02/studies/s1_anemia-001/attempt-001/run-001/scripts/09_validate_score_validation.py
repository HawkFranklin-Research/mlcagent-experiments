import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available top-level keys: {list(protocol)}")
primary_metric = task["primary_metric"]
group_col = protocol.get("group_column")
target_col = targets[0]
if len(targets) != 1:
    raise ValueError("This script expects one target column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

target_set = set(targets)
excluded_set = set(excluded)
candidate = [c for c in train.columns if c not in target_set and c not in excluded_set and c != group_col]
key_like = set()
for c in candidate:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        key_like.add(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        key_like.add(c)
feature_cols = [c for c in candidate if c not in key_like]

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return list(obj.classes_)
    if hasattr(obj, "named_steps"):
        return list(list(obj.named_steps.values())[-1].classes_)
    raise AttributeError("No classes_ found")

specs = [
    ("logistic", inp["07_model_train_logistic_20PTX4M5/model.pkl"]),
    ("alternative", inp["08_model_train_alternative_M8FMHTTA/alternative_model.pkl"]),
]
models = {}
for name, path in specs:
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

first_model = models[specs[0][0]]
positive_label = task.get("positive_label")
if positive_label is None:
    positive_label = get_classes(first_model)[1]
y_val = val[target_col]
y_bin = (y_val == positive_label).astype(int)

rows = []
metric_values = {}
roc_aucs = {}
for name, model in models.items():
    cols = input_cols(model) or feature_cols
    bad = set(cols) & (target_set | excluded_set)
    if bad:
        raise ValueError(f"Model {name} expects target/excluded columns: {bad}")
    X = val[list(cols)]
    proba = model.predict_proba(X)
    classes = get_classes(model)
    pos_idx = classes.index(positive_label)
    proba_pos = proba[:, pos_idx]
    auc = roc_auc_score(y_bin, proba_pos)
    ap = average_precision_score(y_bin, proba_pos)
    fpr, tpr, thr = roc_curve(y_bin, proba_pos)
    j = tpr - fpr
    idx = int(np.argmax(j))
    threshold = float(thr[idx])
    if not np.isfinite(threshold):
        threshold = 0.5
    pred = (proba_pos >= threshold).astype(int)
    bal = balanced_accuracy_score(y_bin, pred)
    brier = brier_score_loss(y_bin, proba_pos)
    roc_aucs[name] = auc
    metric_values[name] = {"roc_auc": auc, "pr_auc": ap, "balanced_accuracy": bal, "brier_score": brier}
    rows.append({"model": name, "metric": primary_metric, "value": metric_values[name][primary_metric], "threshold": np.nan})
    rows.append({"model": name, "metric": "pr_auc", "value": ap, "threshold": np.nan})
    rows.append({"model": name, "metric": "balanced_accuracy", "value": bal, "threshold": threshold})
    rows.append({"model": name, "metric": "brier_score", "value": brier, "threshold": np.nan})

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
df["value"] = df["value"].map(sig4)
df["threshold"] = df["threshold"].map(sig4)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics")

primary_values = {name: metric_values[name][primary_metric] for name in models}
lower_better = {"brier_score", "log_loss", "mae", "mse", "rmse"}
best_name = min(primary_values, key=primary_values.get) if primary_metric in lower_better else max(primary_values, key=primary_values.get)
best_val = primary_values[best_name]
mlca_runtime.record_metric(primary_metric, float(best_val), split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(roc_aucs[best_name]), source="validation_metrics.csv")
print("val", val.shape, "positive", positive_label)
print("best", best_name, primary_metric, round(float(best_val), 4))
