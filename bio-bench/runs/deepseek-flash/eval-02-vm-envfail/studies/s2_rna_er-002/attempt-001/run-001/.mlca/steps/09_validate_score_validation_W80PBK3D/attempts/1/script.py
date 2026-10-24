import json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]
pos_decl = task.get("positive_label")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
target_col = targets[0]
y_val = val[target_col]
feature_cols = [c for c in train.columns if c not in excluded and c not in targets and c != group_col]

model_paths = {
    "logistic": paths["07_model_train_logistic_GBE8KD6W/model.pkl"],
    "alternative": paths["08_model_train_alternative_73DHGKGK/alternative_model.pkl"],
}
models = {}
for nm, p in model_paths.items():
    with open(p, "rb") as f:
        models[nm] = pickle.load(f)

def proba_of(model, frame):
    cols = getattr(model, "feature_names_in_", None)
    cols = list(cols) if cols is not None else feature_cols
    return model.predict_proba(frame[cols])

rows, metric_vals, auc_vals = [], {}, {}
for nm, model in models.items():
    proba = proba_of(model, val)
    classes = list(model.classes_)
    pos = pos_decl if (pos_decl is not None and pos_decl in classes) else classes[1]
    p_pos = proba[:, classes.index(pos)]
    y_bin = (y_val == pos).astype(int).to_numpy()
    auc = float(roc_auc_score(y_bin, p_pos))
    pr = float(average_precision_score(y_bin, p_pos))
    fpr, tpr, thr = roc_curve(y_bin, p_pos)
    t = float(thr[int(np.argmax(tpr - fpr))])
    y_pred = (p_pos >= t).astype(int)
    bacc = float(balanced_accuracy_score(y_bin, y_pred))
    brier = float(brier_score_loss(y_bin, p_pos))
    vals = {"roc_auc": auc, "pr_auc": pr, "balanced_accuracy": bacc, "brier_score": brier}
    metric_vals[nm] = vals[primary_metric]
    auc_vals[nm] = auc
    for m, v in vals.items():
        rows.append({"model": nm, "metric": m, "value": v, "threshold": t})

best_nm = max(metric_vals, key=lambda k: metric_vals[k])
best_val = float(metric_vals[best_nm])
best_auc = float(auc_vals[best_nm])

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
for c in ["value", "threshold"]:
    df[c] = df[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics per model (Youden J threshold from validation).")

for nm in metric_vals:
    mlca_runtime.record_metric(primary_metric, float(metric_vals[nm]), "val")
mlca_runtime.record_metric(primary_metric, best_val, "val")
mlca_runtime.record_fact_candidate("best_model", best_nm, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print("models:", len(models))
print("val rows:", len(val))
print("best_model:", best_nm)
print(f"best_val_{primary_metric}:", round(best_val, 4))
