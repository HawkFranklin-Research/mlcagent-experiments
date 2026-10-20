import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
primary_metric = protocol["task"]["primary_metric"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

val_metrics = pd.read_csv(inputs["10_validate_score_validation_E0F8K8XX/validation_metrics.csv"])
if "split" in val_metrics.columns:
    val_metrics = val_metrics[val_metrics["split"] == "val"]
cand = val_metrics[val_metrics["metric"] == primary_metric]
if len(cand) == 0:
    model_path = inputs["latest_model"]
    model_name = "latest"
else:
    best = cand.sort_values("value", ascending=False).iloc[0]
    model_name = str(best["model"])
    if "baseline" in model_name:
        model_path = inputs["08_model_train_baseline_6RRDC1E6/model.pkl"]
    elif "alternative" in model_name:
        model_path = inputs["09_model_train_alternative_D4ZR7T7K/alternative_model.pkl"]
    else:
        model_path = inputs["latest_model"]
        model_name = "latest"

with open(model_path, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    all_cols = list(train.columns)
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = [c for c in all_cols if c not in drop]
    for c in list(feature_cols):
        s = train[c]
        if (s.dtype == "object" or str(s.dtype) == "category") and s.nunique() == len(s):
            feature_cols.remove(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
            feature_cols.remove(c)

X_test = test[feature_cols]
y_test = test[targets]
preds = np.asarray(model.predict(X_test))
if preds.ndim == 1:
    preds = preds.reshape(-1, len(targets))

f1s, hlosses = [], []
for i, t in enumerate(targets):
    yt = y_test[t].values
    yp = preds[:, i]
    f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    hlosses.append(hamming_loss(yt, yp))
macro_f1 = float(np.mean(f1s))
avg_hamming = float(np.mean(hlosses))
subset_acc = float(np.mean(np.all(preds == y_test.values, axis=1)))

rows = [
    {"model": model_name, "metric": "macro_f1", "value": round(macro_f1, 4)},
    {"model": model_name, "metric": "hamming_loss", "value": round(avg_hamming, 4)},
    {"model": model_name, "metric": "subset_accuracy", "value": round(subset_acc, 4)},
]
for i, t in enumerate(targets):
    rows.append({"model": model_name, "metric": f"macro_f1_{t}", "value": round(float(f1s[i]), 4)})
    rows.append({"model": model_name, "metric": f"hamming_loss_{t}", "value": round(float(hlosses[i]), 4)})

mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary_metric, macro_f1, "test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")

print(f"Test macro_f1: {macro_f1:.4f}")
print(f"Test hamming_loss: {avg_hamming:.4f}")
print(f"Test subset_accuracy: {subset_acc:.4f}")
