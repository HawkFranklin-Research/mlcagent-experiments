import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]
for c in list(feat_cols):
    s = train[c]
    if s.nunique() == len(s) and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
        feat_cols.remove(c)

X_val = val[feat_cols]
y_val = val[targets]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model_paths = {
    "baseline": inputs["07_model_train_baseline_3PEP6N7P/model.pkl"],
    "alternative": inputs["08_model_train_alternative_ACZEHVQ4/alternative_model.pkl"],
    "latest": inputs["latest_model"],
}

rows = []
best_name, best_score = None, -1.0
for name, p in model_paths.items():
    model = load(p)
    y_pred = np.asarray(model.predict(X_val))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    f1s = [f1_score(y_val.iloc[:, i], y_pred[:, i], average="macro") for i in range(len(targets))]
    macro_f1 = float(np.mean(f1s))
    hl = float(hamming_loss(y_val.values, y_pred))
    sa = float(accuracy_score(y_val.values, y_pred))
    rows.append({"model": name, "metric": "macro_f1", "value": macro_f1})
    rows.append({"model": name, "metric": "hamming_loss", "value": hl})
    rows.append({"model": name, "metric": "subset_accuracy", "value": sa})
    for i, t in enumerate(targets):
        rows.append({"model": name, "metric": f"f1_{t}", "value": float(f1s[i])})
    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name

df = pd.DataFrame(rows)
df["value"] = df["value"].round(4)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model")

mlca_runtime.record_metric("macro_f1", best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"models: {list(model_paths)}")
print(f"features: {len(feat_cols)}")
print(f"best: {best_name} macro_f1={best_score:.4f}")
