import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}; available keys: {list(d)}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task)}")
primary = task["primary_metric"]
group_col = protocol.get("group_column")

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
if not hasattr(model, "predict_proba"):
    raise TypeError("latest_model does not provide predict_proba")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])
exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = []
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and "id" in c.lower():
            continue
        feature_cols.append(c)

bad = set(feature_cols) & exclude
if bad:
    raise ValueError(f"Feature list contains excluded columns: {sorted(bad)}")
missing = [c for c in feature_cols if c not in test.columns]
if missing:
    raise KeyError(f"Test is missing model feature columns: {missing}")
if not test.index.is_unique:
    raise ValueError("test index must be unique for row_id")

X_test = test[feature_cols]
proba = np.asarray(model.predict_proba(X_test))
classes = np.asarray(model.classes_)
y_pred = np.asarray(model.predict(X_test))
y_true = test[targets[0]].to_numpy()

def macro_ovr_auc(yt, pr, cls):
    vals = []
    for j, c in enumerate(cls):
        yb = (yt == c)
        if yb.any() and (~yb).any():
            vals.append(roc_auc_score(yb.astype(int), pr[:, j]))
    return float(np.mean(vals)) if vals else np.nan

primary_val = macro_ovr_auc(y_true, proba, classes)
f1_val = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
omitted = [c for c in classes if not np.any(y_true == c)]
if omitted:
    print("classes absent from test:", omitted)

rng = np.random.default_rng(42)
n = len(test)
boot_primary, boot_f1 = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yt, yp, pr = y_true[idx], y_pred[idx], proba[idx, :]
    boot_primary.append(macro_ovr_auc(yt, pr, classes))
    boot_f1.append(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))

def ci(vals):
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

ci_primary, ci_f1 = ci(boot_primary), ci(boot_f1)
metrics_df = pd.DataFrame([
    {"metric": primary, "value": primary_val, "ci_low": ci_primary[0], "ci_high": ci_primary[1]},
    {"metric": "macro_f1", "value": f1_val, "ci_low": ci_f1[0], "ci_high": ci_f1[1]},
])
metrics_df[["value", "ci_low", "ci_high"]] = metrics_df[["value", "ci_low", "ci_high"]].round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, float(primary_val), split="test", ci=ci_primary)
mlca_runtime.record_metric("macro_f1", float(f1_val), split="test", ci=ci_f1)
if primary in ("roc_auc", "roc_auc_ovr"):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(primary_val), source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y_true, "y_pred": y_pred})
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print("test rows:", n, "classes:", len(classes))
print(f"{primary}: {primary_val:.4f} CI [{ci_primary[0]:.4f}, {ci_primary[1]:.4f}]")
print(f"macro_f1: {f1_val:.4f} CI [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
