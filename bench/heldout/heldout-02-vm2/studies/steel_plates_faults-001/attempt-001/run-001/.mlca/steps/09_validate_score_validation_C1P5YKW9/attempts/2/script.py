import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Protocol missing '{key}'. Available keys: {sorted(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

excluded_set = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in excluded_set]
for c in list(feature_cols):
    s = train[c]
    key_like = s.notna().all() and s.nunique() == len(train) and (
        s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    )
    if key_like:
        feature_cols.remove(c)

Y_val = val[targets].to_numpy()


def get_preds(model, frame):
    feat = getattr(model, "feature_names_in_", None)
    if feat is not None:
        X = frame[list(feat)]
    else:
        X = frame[feature_cols]
    p = np.asarray(model.predict(X))
    if p.ndim == 1:
        p = p.reshape(-1, len(targets))
    return p


def metrics(preds, Yv):
    per_f1 = [
        f1_score(Yv[:, i], preds[:, i], average="macro", zero_division=0)
        for i in range(Yv.shape[1])
    ]
    mf1 = float(np.mean(per_f1))
    match = (Yv == preds)
    ham = float(np.mean(~match))
    sub = float(np.mean(np.all(match, axis=1)))
    return mf1, ham, sub


candidates = {
    "baseline": inputs["07_model_train_baseline_HK97MR3S/model.pkl"],
    "alternative": inputs["08_model_train_alternative_2D5YHWB5/alternative_model.pkl"],
}

rows = []
best_name, best_val, higher_better = None, None, True
for name, path in candidates.items():
    with open(path, "rb") as f:
        m = pickle.load(f)
    preds = get_preds(m, val)
    mf1, ham, sub = metrics(preds, Y_val)
    rows.append({"model": name, "metric": "macro_f1", "value": round(mf1, 4)})
    rows.append({"model": name, "metric": "hamming_loss", "value": round(ham, 4)})
    rows.append({"model": name, "metric": "subset_accuracy", "value": round(sub, 4)})
    if best_val is None or mf1 > best_val:
        best_val, best_name = mf1, name

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics per candidate model")

mlca_runtime.record_metric(primary_metric, float(best_val), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_val), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

print(f"features={len(feature_cols)} candidates={len(candidates)}")
print(f"best={best_name} {primary_metric}={best_val:.4f}")
