import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
primary = proto["task"]["primary_metric"]

val = pd.read_csv(inputs["09_validate_score_validation_9862PPB5/validation_metrics.csv"])
if {"model", "metric", "value"}.issubset(val.columns):
    vv = val[val["metric"].astype(str).str.lower() == primary.lower()]
    if vv.empty:
        vv = val
    best_name = str(vv.loc[vv["value"].idxmax(), "model"])
elif "model" in val.columns and primary in val.columns:
    best_name = str(val.loc[val[primary].idxmax(), "model"])
else:
    best_name = "baseline"

if "alt" in best_name.lower() or "latest" in best_name.lower():
    model_path = inputs["08_model_train_alternative_JE7KKHHV/alternative_model.pkl"]
else:
    model_path = inputs["07_model_train_baseline_2N13J4R1/model.pkl"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

test = pd.read_parquet(inputs["test"])
names = getattr(model, "feature_names_in_", None)
if names is None and hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            break
if names is None:
    names = [c for c in test.columns if c not in targets and c not in excluded]

X_test = test[list(names)].copy()
pred = np.asarray(model.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

y_true = test[targets].astype(str).to_numpy()
pred = pred.astype(str)

f1s, hls = [], []
for i in range(len(targets)):
    f1s.append(f1_score(y_true[:, i], pred[:, i], average="macro", zero_division=0))
    hls.append(hamming_loss(y_true[:, i], pred[:, i]))

macro_f1 = float(np.mean(f1s))
hamming = float(np.mean(hls))
subset = float(np.mean(np.all(y_true == pred, axis=1)))

rows = [
    {"model": best_name, "metric": primary, "value": macro_f1},
    {"model": best_name, "metric": "hamming_loss", "value": hamming},
    {"model": best_name, "metric": "subset_accuracy", "value": subset},
]
for i, t in enumerate(targets):
    rows.append({"model": best_name, "metric": f"{t}_macro_f1", "value": f1s[i]})
    rows.append({"model": best_name, "metric": f"{t}_hamming_loss", "value": hls[i]})

metrics = pd.DataFrame(rows)
metrics["value"] = metrics["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(metrics, "test_metrics.csv")
mlca_runtime.record_metric(primary, macro_f1, "test")
mlca_runtime.record_metric("hamming_loss", hamming, "test")
mlca_runtime.record_metric("subset_accuracy", subset, "test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")

print(f"selected={best_name}")
print(f"test_rows={len(test)} features={len(names)}")
print(f"{primary}={macro_f1:.4f} hamming_loss={hamming:.4f} subset_accuracy={subset:.4f}")
