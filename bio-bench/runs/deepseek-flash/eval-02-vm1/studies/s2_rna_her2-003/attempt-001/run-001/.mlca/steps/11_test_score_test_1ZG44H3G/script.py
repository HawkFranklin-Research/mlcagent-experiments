import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score

mlca_runtime.set_seed(42)
rng = np.random.default_rng(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "targets" not in proto:
    raise KeyError(f"targets missing; top-level keys: {list(proto)}")
targets = proto["targets"]
targets = [targets] if isinstance(targets, str) else list(targets)
task = proto.get("task", {})
if "primary_metric" not in task:
    raise KeyError(f"primary_metric missing; task keys: {list(task)}")
primary = task["primary_metric"]
test = pd.read_parquet(inputs["test"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def get_classes(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        last = list(m.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return np.asarray(last.classes_)
    raise AttributeError("no classes_")
classes = get_classes(model)

feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
if feats is None and hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            feats = list(step.feature_names_in_)
            break
if feats is None:
    excluded = proto.get("excluded_columns", [])
    if isinstance(excluded, dict):
        excluded = []
    group = proto.get("group_column", None)
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feats = []
    for c in test.columns:
        if c in drop:
            continue
        s = test[c]
        if s.nunique(dropna=True) == len(s) and (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
            continue
        feats.append(c)
X = test.loc[:, feats].copy()
proba = np.asarray(model.predict_proba(X))

def find_threshold():
    for key in ["10_validate_score_validation_ZV9R75WK/validation_metrics.csv",
                "10_validate_score_validation_ZV9R75WK/validation_metrics.parquet",
                "10_validate_score_validation_ZV9R75WK/manifest.json"]:
        p = inputs.get(key)
        if p is None or not p.exists():
            continue
        try:
            obj = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p) if p.suffix == ".parquet" else json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if "threshold" in str(c).lower():
                    vals = pd.to_numeric(obj[c], errors="coerce").dropna()
                    if len(vals):
                        return float(vals.iloc[0])
            if {"metric", "value"}.issubset(obj.columns):
                m = obj["metric"].astype(str).str.lower().str.contains("threshold|thr|cutoff")
                if m.any():
                    vals = pd.to_numeric(obj.loc[m, "value"], errors="coerce").dropna()
                    if len(vals):
                        return float(vals.iloc[0])
        else:
            def rec(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if "threshold" in str(k).lower() or str(k).lower() in ("thr", "cutoff"):
                            try:
                                return float(v)
                            except Exception:
                                pass
                        r = rec(v)
                        if r is not None:
                            return r
                elif isinstance(o, list):
                    for v in o:
                        r = rec(v)
                        if r is not None:
                            return r
                return None
            r = rec(obj)
            if r is not None:
                return r
    for attr in ("threshold_", "threshold"):
        if hasattr(model, attr):
            try:
                return float(getattr(model, attr))
            except Exception:
                pass
    return 0.5
threshold = find_threshold()

y = test[targets[0]].to_numpy()
if len(classes) == 2:
    pos = task.get("positive_label", classes[1])
    if pos is None:
        pos = classes[1]
    pos_idx = list(classes).index(pos)
    neg = classes[1] if pos == classes[0] else classes[0]
    ppos = proba[:, pos_idx]
    ybin = (y == pos).astype(int)
    ypred = np.where(ppos >= threshold, pos, neg)
    roc = roc_auc_score(ybin, ppos) if len(np.unique(ybin)) > 1 else np.nan
    pr = average_precision_score(ybin, ppos) if ybin.sum() > 0 else np.nan
    bal = balanced_accuracy_score(y, ypred)
    brier = brier_score_loss(ybin, ppos)
    tn, fp, fn, tp = confusion_matrix(ybin, (ypred == pos).astype(int), labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    vals = []
    n = len(ybin)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        yb = ybin[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, ppos[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    rows = [(primary, roc, lo, hi), ("pr_auc", pr, np.nan, np.nan),
            ("balanced_accuracy", bal, np.nan, np.nan), ("brier_score", brier, np.nan, np.nan),
            ("sensitivity", sens, np.nan, np.nan), ("specificity", spec, np.nan, np.nan)]
else:
    ypred = model.predict(X)
    roc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    rows = [(primary, roc, np.nan, np.nan), ("macro_f1", f1_score(y, ypred, average="macro"), np.nan, np.nan)]
    lo = hi = np.nan

def r4(x):
    try:
        return np.nan if pd.isna(x) else float(f"{float(x):.4g}")
    except Exception:
        return x
metrics = pd.DataFrame([(m, r4(v), r4(a), r4(b)) for m, v, a, b in rows],
                       columns=["metric", "value", "ci_low", "ci_high"])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary, float(roc), "test",
                           ci=(float(lo), float(hi)) if np.isfinite(lo) and np.isfinite(hi) else None)
if len(classes) == 2 and np.isfinite(roc):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(roc), source="test_metrics.csv")

pred = pd.DataFrame({"row_id": test.index, "partition": "test"})
if len(targets) == 1:
    pred["y_true"] = test[targets[0]].to_numpy()
    pred["y_pred"] = ypred
else:
    for t in targets:
        pred[f"y_true__{t}"] = test[t].to_numpy()
        pred[f"y_pred__{t}"] = ypred
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")
print(f"test rows: {len(test)}")
print(f"{primary}: {roc:.4f} CI [{lo:.4f}, {hi:.4f}] threshold={threshold}")
