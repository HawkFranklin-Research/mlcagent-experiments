import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
available = sorted(protocol.keys())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {available}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task.keys())}")
primary = str(task["primary_metric"])

train = pd.read_parquet(inputs["train"])
base_cols = [c for c in train.columns if c not in set(targets) | set(excluded)]
if group is not None and group in base_cols:
    base_cols.remove(group)

features = []
for c in base_cols:
    s = train[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    if is_text and s.nunique(dropna=False) == len(s):
        continue
    features.append(c)

val = pd.read_csv(inputs["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"])
if "metric" not in val.columns or "value" not in val.columns:
    raise KeyError(f"Validation metrics need metric/value columns; got {list(val.columns)}")
sub = val[val["metric"].astype(str) == primary].copy()
if sub.empty:
    raise ValueError(f"No validation rows for declared primary metric {primary!r}")

lower_is_better = any(w in primary.lower() for w in ("loss", "error", "hamming"))
best = sub.sort_values("value", ascending=lower_is_better).iloc[0]
model_name = str(best["model"]) if "model" in sub.columns else "selected"

s = model_name.lower()
if "alt" in s or "latest" in s or "08_" in s:
    model_path = inputs["latest_model"]
elif "base" in s or "07_" in s or s == "model.pkl":
    model_path = inputs["07_model_train_baseline_HK97MR3S/model.pkl"]
else:
    model_path = inputs["07_model_train_baseline_HK97MR3S/model.pkl"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

expected = list(getattr(model, "feature_names_in_", [])) if hasattr(model, "feature_names_in_") else []
if not expected:
    expected = features

bad = set(expected) & (set(targets) | set(excluded) | ({group} if group is not None else set()))
if bad:
    raise ValueError(f"Model input includes target/excluded/group columns: {sorted(bad)}")

test = pd.read_parquet(inputs["test"])
missing = [c for c in expected if c not in test.columns]
if missing:
    raise KeyError(f"Missing test feature columns: {missing}")

X = test[expected].copy()
pred = np.asarray(model.predict(X))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets):
    raise ValueError(f"Prediction columns {pred.shape[1]} != targets {len(targets)}")

f1s = []
hams = []
for j, t in enumerate(targets):
    yt = test[t].to_numpy()
    yp = pred[:, j]
    f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
    hams.append(float(hamming_loss(yt, yp)))

macro_f1 = float(np.mean(f1s))
hamming_avg = float(np.mean(hams))
subset_acc = float(np.mean(np.all(pred == test[targets].to_numpy(), axis=1)))

rows = [
    {"model": model_name, "metric": primary, "value": macro_f1},
    {"model": model_name, "metric": "hamming_loss", "value": hamming_avg},
    {"model": model_name, "metric": "subset_accuracy", "value": subset_acc},
]
for t, v in zip(targets, f1s):
    rows.append({"model": model_name, "metric": f"macro_f1_{t}", "value": float(v)})

metrics = pd.DataFrame(rows)
metrics["value"] = metrics["value"].map(lambda z: float(f"{z:.4g}"))
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics for validation-selected model")
mlca_runtime.record_metric(primary, macro_f1, "test")
mlca_runtime.record_metric("hamming_loss", hamming_avg, "test")
mlca_runtime.record_metric("subset_accuracy", subset_acc, "test")
mlca_runtime.record_fact_candidate(
    "test_score",
    float(metrics.loc[metrics["metric"] == primary, "value"].iloc[0]),
    source="test_metrics.csv",
)

print(f"selected={model_name}")
print(f"test_rows={len(test)}")
print(f"{primary}={macro_f1:.4f}")
