import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "task" not in proto or not isinstance(proto["task"], dict):
    raise KeyError(f"Missing task; available top-level keys: {list(proto.keys())}")
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available top-level keys: {list(proto.keys())}")
primary = task["primary_metric"]
if "targets" not in proto or not proto["targets"]:
    raise KeyError(f"Missing targets; available top-level keys: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")
if len(targets) != 1:
    raise ValueError("This script expects exactly one target")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def is_key_like(s):
    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return bool(s.is_unique)
    return False

forbidden = set(targets) | set(excluded)
if group is not None:
    forbidden.add(group)
feat_cols = [c for c in train.columns if c not in forbidden and not is_key_like(train[c])]

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

def model_cols(m, frame, cols):
    if hasattr(m, "feature_names_in_"):
        cols = list(m.feature_names_in_)
    bad = set(cols) & forbidden
    if bad:
        raise ValueError(f"Model expects forbidden columns: {bad}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing model columns: {missing}")
    return frame[cols]

X_test = model_cols(model, test, feat_cols)
y = test[target].to_numpy()
pred = model.predict(X_test)
P = model.predict_proba(X_test)
classes = getattr(model, "classes_", None)
if classes is None and hasattr(model, "named_steps"):
    last = model.named_steps[list(model.named_steps)[-1]]
    classes = getattr(last, "classes_", None)
if classes is None:
    raise AttributeError("Model has no classes_")
classes = np.asarray(classes)

def macro_ovr_auc(yy, PP, cls):
    vals = []
    for j, c in enumerate(cls):
        yb = (yy == c).astype(int)
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, PP[:, j]))
    return float(np.mean(vals)) if vals else float("nan")

def macro_f1(yy, pp, cls):
    return float(f1_score(yy, pp, average="macro", labels=cls, zero_division=0))

auc = macro_ovr_auc(y, P, classes)
f1 = macro_f1(y, pred, classes)
missing = [c for c in classes if not (y == c).any()]
if missing:
    print("classes absent from test", list(map(str, missing)))

rng = np.random.default_rng(42)
n = len(y)
auc_b, f1_b = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    auc_b.append(macro_ovr_auc(y[idx], P[idx], classes))
    f1_b.append(macro_f1(y[idx], pred[idx], classes))
auc_ci = tuple(np.nanpercentile(auc_b, [2.5, 97.5]))
f1_ci = tuple(np.nanpercentile(f1_b, [2.5, 97.5]))

if primary in ("roc_auc_ovr", "roc_auc"):
    pv, pci = auc, auc_ci
elif primary in ("f1_macro", "f1"):
    pv, pci = f1, f1_ci
else:
    raise ValueError(f"Unsupported primary_metric: {primary}")

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

rows = [{"metric": primary, "value": sig4(pv), "ci_low": sig4(pci[0]), "ci_high": sig4(pci[1])}]
if primary not in ("macro_f1", "f1_macro"):
    rows.append({"metric": "macro_f1", "value": sig4(f1), "ci_low": sig4(f1_ci[0]), "ci_high": sig4(f1_ci[1])})
if primary not in ("roc_auc_ovr", "roc_auc"):
    rows.append({"metric": "roc_auc_ovr", "value": sig4(auc), "ci_low": sig4(auc_ci[0]), "ci_high": sig4(auc_ci[1])})
metrics = pd.DataFrame(rows, columns=["metric", "value", "ci_low", "ci_high"])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary, float(pv), "test", ci=(float(pci[0]), float(pci[1])))
if primary not in ("macro_f1", "f1_macro"):
    mlca_runtime.record_metric("macro_f1", float(f1), "test", ci=(float(f1_ci[0]), float(f1_ci[1])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index, "partition": "test"})
out[f"y_true__{target}"] = y
out[f"y_pred__{target}"] = pred
for j, c in enumerate(classes):
    out[f"proba__{c}"] = P[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print("test rows", len(test), "primary", primary, "value", round(float(pv), 4))
print("classes", list(map(str, classes)))
