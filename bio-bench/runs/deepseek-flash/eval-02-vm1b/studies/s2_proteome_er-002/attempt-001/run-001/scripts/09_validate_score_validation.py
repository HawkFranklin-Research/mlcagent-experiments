import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]; metric_name = task["primary_metric"]
targets = protocol["targets"]
excluded = set(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"])

drop = set(targets) | excluded | ({group} if group else set())
feat = [c for c in train.columns if c not in drop]
keylike = [c for c in feat
           if (not pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_integer_dtype(train[c]))
           and train[c].nunique(dropna=False) == len(train)]
feat = [c for c in feat if c not in keylike]

models = {
    "07_model_train_logistic_RQ5H03SW": pickle.load(open(inp["07_model_train_logistic_RQ5H03SW/model.pkl"], "rb")),
    "08_model_train_alternative_7HJK4116": pickle.load(open(inp["08_model_train_alternative_7HJK4116/alternative_model.pkl"], "rb")),
}

y_val = val[targets[0]]
rows = []; results = {}
for name, model in models.items():
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feat
    Xv = val[list(cols)]
    proba = model.predict_proba(Xv)
    classes = list(model.classes_)
    pos = classes[1]
    p = proba[:, classes.index(pos)]
    yb = (y_val == pos).astype(int)
    roc = roc_auc_score(yb, p); pr = average_precision_score(yb, p); br = brier_score_loss(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    bi = int(np.argmax(tpr - fpr)); ythr = float(thr[bi])
    ba = balanced_accuracy_score(yb, (p >= ythr).astype(int))
    results[name] = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": ba, "brier_score": br,
                     "threshold": ythr, "pos_label": pos}
    for mn, mv, tv in [("roc_auc", roc, np.nan), ("pr_auc", pr, np.nan),
                       ("balanced_accuracy", ba, ythr), ("brier_score", br, np.nan)]:
        rows.append({"model": name, "metric": mn, "value": mv, "threshold": tv})

table = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(table, "validation_metrics.csv", "Validation metrics per model (binary)")

lower_better = {"brier_score", "log_loss", "mae", "rmse", "mse"}
direction = min if metric_name in lower_better else max
best_model = direction(results, key=lambda k: results[k][metric_name])
for name in results:
    mlca_runtime.record_metric(metric_name, float(results[name][metric_name]), split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(results[best_model]["roc_auc"]),
                                   source="validation_metrics.csv")
print(f"features={len(feat)} models={len(models)}")
print(f"primary={metric_name} best={best_model} val={results[best_model][metric_name]:.4f}")
print(f"youden_thresholds={ {k: round(v['threshold'],4) for k,v in results.items()} }")
