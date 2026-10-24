import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def get_key(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {list(d)}")
    return d[k]

protocol = json.loads(inputs["_protocol"].read_text())
targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group_col = get_key(protocol, "group_column")
task = get_key(protocol, "task")
primary = get_key(task, "primary_metric")
target = targets[0]

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def get_feature_names(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

feature_cols = get_feature_names(model)
if feature_cols is None:
    forbidden = set(targets) | set(excluded)
    if group_col is not None:
        forbidden.add(group_col)
    cand = [c for c in train.columns if c not in forbidden]
    key_like = []
    for c in cand:
        s = train[c]
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            if s.nunique(dropna=False) == len(train) and s.notna().all():
                key_like.append(c)
        elif pd.api.types.is_integer_dtype(s):
            name = c.lower()
            if s.nunique(dropna=False) == len(train) and any(k in name for k in ("id", "index", "code")):
                key_like.append(c)
    feature_cols = [c for c in cand if c not in key_like and c in test.columns]

forbidden = set(targets) | set(excluded)
if group_col is not None:
    forbidden.add(group_col)
if forbidden & set(feature_cols):
    raise ValueError("Target/excluded/group column reached feature list")

missing = [c for c in feature_cols if c not in test.columns]
if missing:
    raise KeyError(f"Missing test feature columns: {missing}")

X_test = test[feature_cols]
y_true = test[target].to_numpy()
proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

if hasattr(model, "classes_"):
    classes = np.asarray(model.classes_)
elif hasattr(model, "named_steps"):
    classes = np.asarray(list(model.named_steps.values())[-1].classes_)
else:
    raise AttributeError("Model has no classes_")

def multiclass_auc(yt, pr, cls):
    present = [j for j, c in enumerate(cls) if np.any(yt == c)]
    if len(present) < 2:
        return np.nan
    vals = []
    for j in present:
        yb = (yt == cls[j]).astype(int)
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, pr[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def auc_metric(yt, pr, yp, cls):
    return multiclass_auc(yt, pr, cls)

def f1_metric(yt, pr, yp, cls):
    return f1_score(yt, yp, average="macro", labels=cls, zero_division=0)

def bootstrap_ci(metric_fn, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], proba[idx], y_pred[idx], classes))
        except Exception:
            vals.append(np.nan)
    vals = np.asarray(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan, np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

auc = multiclass_auc(y_true, proba, classes)
f1 = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
auc_lo, auc_hi = bootstrap_ci(auc_metric)
f1_lo, f1_hi = bootstrap_ci(f1_metric)

metrics = pd.DataFrame([
    {"metric": primary, "value": round(float(auc), 4), "ci_low": round(float(auc_lo), 4), "ci_high": round(float(auc_hi), 4)},
    {"metric": "macro_f1", "value": round(float(f1), 4), "ci_low": round(float(f1_lo), 4), "ci_high": round(float(f1_hi), 4)},
])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary, float(auc), split="test", ci=(float(auc_lo), float(auc_hi)))
mlca_runtime.record_metric("macro_f1", float(f1), split="test", ci=(float(f1_lo), float(f1_hi)))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

pred_frame = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    f"y_true__{target}": y_true,
    f"y_pred__{target}": y_pred,
})
for j, cls in enumerate(classes):
    pred_frame[f"proba__{cls}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_frame, "test_predictions.parquet")

print(f"test rows: {len(test)}")
print(f"features: {len(feature_cols)}")
print(f"{primary}: {auc:.4f} CI [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"macro_f1: {f1:.4f} CI [{f1_lo:.4f}, {f1_hi:.4f}]")
