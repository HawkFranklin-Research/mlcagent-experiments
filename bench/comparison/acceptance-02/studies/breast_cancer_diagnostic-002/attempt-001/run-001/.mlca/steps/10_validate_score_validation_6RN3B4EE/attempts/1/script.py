import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y_val = val[target].values

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
X_val = val[feature_cols]

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_a = load(inputs["08_model_train_logistic_71AKM31B/model.pkl"])
model_b = load(inputs["09_model_train_alternative_QMPABASH/alternative_model.pkl"])
preprocessor = load(inputs["06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl"])
selector = load(inputs["07_features_select_features_B35TTT17/selector.pkl"])

def predict_proba(model, X):
    if isinstance(model, Pipeline) and hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    Xp = preprocessor.transform(X)
    Xs = selector.transform(Xp)
    return model.predict_proba(Xs)

def evaluate(model, X, y, positive_label=None):
    y = np.asarray(y)
    proba = predict_proba(model, X)
    classes = model.classes_ if hasattr(model, "classes_") else model.steps[-1][1].classes_
    if positive_label is None:
        pos_idx = 1
        pos_label = classes[1]
    else:
        pos_idx = list(classes).index(positive_label)
        pos_label = positive_label
    y_bin = (y == pos_label).astype(int)
    p_pos = proba[:, pos_idx]
    roc_auc = roc_auc_score(y_bin, p_pos)
    pr_auc = average_precision_score(y_bin, p_pos)
    fpr, tpr, thresholds = roc_curve(y_bin, p_pos)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (p_pos >= best_thresh).astype(int)
    bal_acc = balanced_accuracy_score(y_bin, y_pred)
    brier = brier_score_loss(y_bin, p_pos)
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "balanced_accuracy": bal_acc,
            "brier": brier, "threshold": best_thresh, "pos_label": pos_label}

pos_label = protocol.get("positive_label") or task.get("positive_label")
res_a = evaluate(model_a, X_val, y_val, positive_label=pos_label)
res_b = evaluate(model_b, X_val, y_val, positive_label=pos_label)

rows = []
for name, res in [("logistic", res_a), ("alternative", res_b)]:
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier"]:
        thr = res["threshold"] if metric == "balanced_accuracy" else np.nan
        rows.append({"model": name, "metric": metric, "value": res[metric], "threshold": thr})
df = pd.DataFrame(rows)
df["value"] = df["value"].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
df["threshold"] = df["threshold"].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models")

mlca_runtime.record_metric(primary_metric, res_a[primary_metric], split="val")
mlca_runtime.record_metric(primary_metric, res_b[primary_metric], split="val")

if res_a[primary_metric] >= res_b[primary_metric]:
    best_name, best_res = "logistic", res_a
else:
    best_name, best_res = "alternative", res_b
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_res["roc_auc"], source="validation_metrics.csv")

print(f"roc_auc logistic={res_a['roc_auc']:.4f} alternative={res_b['roc_auc']:.4f}")
print(f"best_model={best_name}")
