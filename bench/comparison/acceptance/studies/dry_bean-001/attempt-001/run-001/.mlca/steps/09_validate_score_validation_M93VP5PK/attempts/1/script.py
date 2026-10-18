import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

task = proto["task"]
primary_metric = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]
for c in list(feature_cols):
    s = train[c]
    if (s.dtype == object) or np.issubdtype(s.dtype, np.integer):
        if s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)

X_val = val[feature_cols]
y_val = val[targets[0]]

model_paths = {
    "logistic": inp["07_model_train_logistic_D2WYGTBM/model.pkl"],
    "alternative": inp["08_model_train_alternative_5QQJK8QV/alternative_model.pkl"],
}

rows = []
results = {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    proba = model.predict_proba(X_val)
    classes = model.classes_
    pred = model.predict(X_val)
    auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    f1 = f1_score(y_val, pred, average="macro")
    results[name] = auc
    rows.append({"model": name, "metric": primary_metric, "value": auc, "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": f1, "threshold": np.nan})

best_model = max(results, key=results.get)
best_auc = results[best_model]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")

for name, auc in results.items():
    mlca_runtime.record_metric(primary_metric, float(auc), "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_auc), source="validation_metrics.csv")

print(f"features={len(feature_cols)} train_rows={len(train)} val_rows={len(val)}")
print(f"best_model={best_model} {primary_metric}={best_auc:.4f}")
