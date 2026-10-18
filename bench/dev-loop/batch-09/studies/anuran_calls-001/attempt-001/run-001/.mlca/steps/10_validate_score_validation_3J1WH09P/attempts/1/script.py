import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(task["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)
primary = task["primary_metric"]

val = pd.read_parquet(inp["val"])
feature_cols = [c for c in val.columns if c not in targets and c not in excluded and c != group]

def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def evaluate(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    names = input_names(model)
    X = val[names] if names else val[feature_cols]
    pred = model.predict(X)
    if hasattr(pred, "to_numpy"):
        pred = pred.to_numpy()
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    Y = val[targets].to_numpy()
    per_f1 = [float(f1_score(Y[:, i], pred[:, i], average="macro", zero_division=0))
              for i in range(len(targets))]
    per_hl = [float(hamming_loss(Y[:, i], pred[:, i])) for i in range(len(targets))]
    return {
        "macro_f1": float(np.mean(per_f1)),
        "hamming_loss_avg": float(np.mean(per_hl)),
        "subset_accuracy": float(np.mean(np.all(pred == Y, axis=1))),
    }

candidates = {
    "baseline": inp["08_model_train_baseline_RDZRN5AQ/model.pkl"],
    "alternative": inp["09_model_train_alternative_J8YMJGXB/alternative_model.pkl"],
}
results = {name: evaluate(path) for name, path in candidates.items()}

# primary_metric = macro_f1: higher is better
best_name = max(results, key=lambda k: results[k][primary])
best_score = results[best_name][primary]

rows = [
    {"model": m, "metric": k, "value": float(f"{v:.4g}")}
    for m, r in results.items() for k, v in r.items()
]
mlca_runtime.save_table(pd.DataFrame(rows), "validation_metrics.csv")

mlca_runtime.record_metric(str(primary), float(best_score), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")

print("val rows:", len(val), "| candidates:", list(results))
print("best:", best_name, "| " + str(primary) + ":", round(float(best_score), 4))
