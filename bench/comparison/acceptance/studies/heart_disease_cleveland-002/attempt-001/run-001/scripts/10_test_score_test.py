import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, roc_curve
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}") from e
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    name = str(s.name).lower()
    if s.dtype.kind in "OUS" or str(s.dtype) in ("category", "string"):
        return True
    return any(t in name for t in ["id", "key", "code", "uuid", "guid", "patient", "record"])

blocked = set(targets + excluded + ([group] if group else []))
features = [c for c in train.columns if c not in blocked and not key_like(train[c])]

def load_json(p):
    return json.loads(p.read_text())

m7 = load_json(inp["07_model_train_logistic_8ET5BK8M/manifest.json"])
m8 = load_json(inp["08_model_train_alternative_TE7YJ5NR/manifest.json"])
vman = load_json(inp["09_validate_score_validation_JHNMD07Q/manifest.json"])
vdf = pd.read_csv(inp["09_validate_score_validation_JHNMD07Q/validation_metrics.csv"])

def col_like(df, opts):
    for c in df.columns:
        if str(c).lower() in opts:
            return c
    return None

lower = any(t in primary.lower() for t in ["brier", "loss", "error", "mae", "mse", "rmse"])
mcol = col_like(vdf, {"metric", "name", "metric_name"})
vcol = col_like(vdf, {"value", "score", "mean"})
ccol = col_like(vdf, {"model", "candidate", "model_name", "estimator", "pipeline", "run", "variant"})
model_path, man = None, None
if mcol and vcol and ccol:
    sub = vdf[vdf[mcol].astype(str) == primary]
    best = None
    for _, r in sub.iterrows():
        cand = str(r[ccol]).lower()
        if "logistic" in cand or "07" in cand:
            p, m = inp["07_model_train_logistic_8ET5BK8M/model.pkl"], m7
        elif "alternative" in cand or "08" in cand:
            p, m = inp["08_model_train_alternative_TE7YJ5NR/alternative_model.pkl"], m8
        else:
            continue
        try:
            val = float(r[vcol])
        except Exception:
            continue
        if best is None or (val < best[0] if lower else val > best[0]):
            best = (val, p, m)
    if best:
        _, model_path, man = best
if model_path is None:
    model_path = inp["latest_model"]
    man = m7 if "logistic" in str(model_path).lower() else m8

with open(model_path, "rb") as f:
    model = pickle.load(f)

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        for v in o.values():
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_thr(v)
            if r is not None:
                return r
    return None

thr = find_thr(man)
if thr is None:
    thr = find_thr(vman)

classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model.steps[-1][1].classes_)
pos = task.get("positive_label")
if pos is None:
    pos = classes[1]
if not np.any(classes == pos):
    raise ValueError(f"positive label {pos} not in model classes")
pos_idx = int(np.where(classes == pos)[0][0])

if thr is None:
    val = pd.read_parquet(inp["val"])
    pv = model.predict_proba(val[features].copy())[:, pos_idx]
    yv = (val[target].to_numpy() == pos).astype(int)
    try:
        fpr, tpr, ths = roc_curve(yv, pv)
        thr = float(ths[int(np.nanargmax(tpr + (1 - fpr)))])
    except Exception:
        thr = 0.5

p = model.predict_proba(test[features].copy())[:, pos_idx]
y = test[target].to_numpy()
y_bin = (y == pos).astype(int)

def sens_spec(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)

def one(metric, y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    try:
        if metric == "roc_auc":
            return roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
        if metric == "pr_auc":
            return average_precision_score(y_true, prob)
        if metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, pred)
        if metric == "brier_score":
            return brier_score_loss(y_true, prob)
        if metric == "sensitivity":
            return sens_spec(y_true, pred)[0]
        if metric == "specificity":
            return sens_spec(y_true, pred)[1]
    except Exception:
        return np.nan
    return np.nan

base = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
names = base if primary in base else [primary] + base
rng = np.random.default_rng(42)
n = len(y_bin)
boot = {m: [] for m in names}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    for m in names:
        boot[m].append(one(m, y_bin[idx], p[idx], thr))

rows = []
for m in names:
    vals = np.asarray(boot[m], dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    rows.append({"metric": m, "value": one(m, y_bin, p, thr), "ci_low": lo, "ci_high": hi})

def sig4(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

out = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    out[c] = out[c].map(sig4)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

prow = out.loc[out["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prow["value"]), "test", (float(prow["ci_low"]), float(prow["ci_high"])))
auc = float(out.loc[out["metric"] == "roc_auc", "value"].iloc[0])
mlca_runtime.record_fact_candidate("test_roc_auc", auc, source="test_metrics.csv")

print(f"test n={len(test)} features={len(features)}")
print(f"{primary}={prow['value']:.4f} CI [{prow['ci_low']:.4f}, {prow['ci_high']:.4f}]")
print("saved test_metrics.csv")
