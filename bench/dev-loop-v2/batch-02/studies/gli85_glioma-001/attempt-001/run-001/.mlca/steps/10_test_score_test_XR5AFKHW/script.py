import mlca_runtime
import json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing primary_metric; available task keys: {list(task)}")
primary_metric = task["primary_metric"]
targets = list(protocol["targets"])
if not targets:
    raise ValueError("No targets declared")
target = targets[0]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def get_classes(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise ValueError("Model has no classes_")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | excluded | ({group} if group else set())
fallback_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (
        not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        continue
    fallback_cols.append(c)

def get_X(df, model):
    fn = getattr(model, "feature_names_in_", None)
    cols = list(fn) if fn is not None else fallback_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}")
    return df.loc[:, cols], cols

def macro_ovr_auc(y, proba, classes):
    vals = []
    present = set(np.unique(y))
    for j, c in enumerate(classes):
        if c in present and len(np.unique(y == c)) > 1:
            vals.append(roc_auc_score((y == c).astype(int), proba[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def val_score(m):
    Xv, _ = get_X(val, m)
    p = m.predict_proba(Xv)
    c = get_classes(m)
    if task.get("type") == "binary":
        pos = task.get("positive_label", c[1])
        j = list(c).index(pos)
        return roc_auc_score((val[target].to_numpy() == pos).astype(int), p[:, j])
    return macro_ovr_auc(val[target].to_numpy(), p, c)

cand_keys = [k for k in inputs if "model" in k.lower() and not any(
    x in k.lower() for x in ("manifest", "preprocess", "selector")
)]
seen, cands = set(), []
for k in cand_keys:
    p = inputs[k].resolve()
    if p in seen:
        continue
    seen.add(p)
    m = load_pickle(inputs[k])
    if hasattr(m, "predict_proba"):
        cands.append((k, m))
if not cands:
    raise ValueError("No model candidates with predict_proba found")
val = pd.read_parquet(inputs["val"])
scores = [(val_score(m), k, m) for k, m in cands]
scores = [s for s in scores if np.isfinite(s[0])]
if not scores:
    raise ValueError("No finite validation scores")
minimize = any(x in primary_metric.lower() for x in ("brier", "log_loss", "mae", "mse", "error"))
_, selected_key, model = min(scores, key=lambda z: z[0]) if minimize else max(scores, key=lambda z: z[0])

val_X, feature_cols = get_X(val, model)
val_proba = model.predict_proba(val_X)
classes = get_classes(model)
if task.get("type") == "binary":
    pos = task.get("positive_label", classes[1])
    pos_idx = list(classes).index(pos)
    ind = (val[target].to_numpy() == pos).astype(int)
    fpr, tpr, thr = roc_curve(ind, val_proba[:, pos_idx])
    ok = np.isfinite(thr)
    threshold = float(thr[ok][np.argmax(tpr[ok] - fpr[ok])]) if ok.any() else 0.5
else:
    threshold = None

test = pd.read_parquet(inputs["test"])
X_test, feature_cols = get_X(test, model)
proba = model.predict_proba(X_test)
y_true = test[target].to_numpy()
y_pred = model.predict(X_test)
classes = get_classes(model)

def calc_metrics(y, yp, pr, cls, ttype, th):
    if ttype == "binary":
        pos = task.get("positive_label", cls[1])
        j = list(cls).index(pos)
        ind = (y == pos).astype(int)
        p = pr[:, j]
        pred = p >= th
        tp = int(((ind == 1) & pred).sum()); fn = int(((ind == 1) & ~pred).sum())
        tn = int(((ind == 0) & ~pred).sum()); fp = int(((ind == 0) & pred).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        return {
            "roc_auc": roc_auc_score(ind, p) if len(np.unique(ind)) > 1 else np.nan,
            "pr_auc": average_precision_score(ind, p),
            "balanced_accuracy": (sens + spec) / 2,
            "brier_score": float(np.mean((p - ind) ** 2)),
            "sensitivity": sens,
            "specificity": spec,
        }
    return {
        "roc_auc_ovr": macro_ovr_auc(y, pr, cls),
        "f1_macro": f1_score(y, yp, average="macro"),
    }

metrics = calc_metrics(y_true, y_pred, proba, classes, task.get("type"), threshold)
metric_names = list(metrics)
rng = np.random.default_rng(42)
n = len(y_true)
boot_vals = {m: [] for m in metric_names}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, ypb, pb = y_true[idx], y_pred[idx], proba[idx]
    if len(np.unique(yb)) < 2:
        continue
    mb = calc_metrics(yb, ypb, pb, classes, task.get("type"), threshold)
    for m in metric_names:
        if np.isfinite(mb[m]):
            boot_vals[m].append(mb[m])

def r4(x):
    try:
        x = float(x)
        return float(f"{x:.4g}") if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

rows, ci = [], {}
for m in metric_names:
    arr = np.asarray(boot_vals[m])
    ci[m] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if arr.size else (np.nan, np.nan)
    rows.append({"metric": m, "value": r4(metrics[m]), "ci_low": r4(ci[m][0]), "ci_high": r4(ci[m][1])})
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pm_key = primary_metric if primary_metric in metrics else primary_metric.strip().lower().replace(" ", "_")
if pm_key in metrics:
    mlca_runtime.record_metric(primary_metric, float(metrics[pm_key]), "test", ci=(float(ci[pm_key][0]), float(ci[pm_key][1])))
if pm_key == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(metrics["roc_auc"]), source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index.to_numpy(), "partition": "test"})
out["y_true"] = y_true
out["y_pred"] = y_pred
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print(f"selected={selected_key} test_rows={len(test)} features={len(feature_cols)}")
print(f"{primary_metric}={metrics.get(pm_key, np.nan):.4g} ci=({ci.get(pm_key, (np.nan, np.nan))[0]:.4g},{ci.get(pm_key, (np.nan, np.nan))[1]:.4g})")
