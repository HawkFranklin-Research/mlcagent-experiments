import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(Path(inp["_protocol"]).read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])
drop = set(targets + excluded + ([group] if group else []))
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)
Xval = val[feature_cols]
Xtest = test[feature_cols]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

prep = load_pkl(inp["05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_HGTRMBKM/selector.pkl"])

def final_est(o):
    while isinstance(o, Pipeline):
        o = o.steps[-1][1]
    return o

def build_pipe(m, Xraw):
    if isinstance(m, Pipeline):
        try:
            m.predict_proba(Xraw.head(5))
            return m
        except Exception:
            pass
    use_prep = prep is not None
    if isinstance(sel, Pipeline):
        keys = [k.lower() for k in sel.named_steps]
        if any(("prep" in k or "column" in k or "transform" in k) for k in keys):
            use_prep = False
    steps = []
    if use_prep:
        steps.append(("prep", prep))
    if sel is not None:
        steps.append(("sel", sel))
    steps.append(("model", m))
    return Pipeline(steps)

def pos_prob(pipe, X, positive=None):
    est = final_est(pipe)
    classes = list(est.classes_)
    pos = positive if positive in classes else classes[1]
    p = pipe.predict_proba(X)[:, classes.index(pos)]
    return p, pos, classes

def find_key(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in keys:
                return v
            r = find_key(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, keys)
            if r is not None:
                return r
    return None

m9 = json.loads(Path(inp["09_validate_score_validation_JDK2E2Q0/manifest.json"]).read_text())
saved_thr = find_key(m9, {"threshold", "decision_threshold", "frozen_threshold",
                          "best_threshold", "optimal_threshold", "val_threshold"})
try:
    saved_thr = float(saved_thr) if saved_thr is not None else None
except Exception:
    saved_thr = None

cand = [
    ("07_logistic", inp["07_model_train_logistic_X3NX87MM/model.pkl"]),
    ("08_alternative", inp["08_model_train_alternative_5Z1FXR5Y/alternative_model.pkl"]),
    ("latest", inp["latest_model"]),
]
best = None
seen = set()
for name, p in cand:
    key = str(p)
    if key in seen:
        continue
    seen.add(key)
    m = load_pkl(p)
    pipe = build_pipe(m, Xval)
    pv, posv, _ = pos_prob(pipe, Xval)
    yv = (val[targets[0]].to_numpy() == posv).astype(int)
    score = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else -np.inf
    if best is None or score > best[0]:
        best = (score, name, pipe, posv)

_, sel_name, pipe, positive = best
p_test, positive, classes = pos_prob(pipe, Xtest, positive)

if saved_thr is None or not np.isfinite(saved_thr):
    pv, posv, _ = pos_prob(pipe, Xval, positive)
    yv = (val[targets[0]].to_numpy() == posv).astype(int)
    fpr, tpr, thr = roc_curve(yv, pv)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    threshold = float(thr[idx]) if np.isfinite(thr[idx]) else 0.5
else:
    threshold = saved_thr

metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score",
                "sensitivity", "specificity"]

def metric_values(y, p, t):
    pred = (p >= t).astype(int)
    out = {}
    out["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    out["pr_auc"] = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan
    out["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    out["brier_score"] = brier_score_loss(y, p)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return out

y_test = (test[targets[0]].to_numpy() == positive).astype(int)
point = metric_values(y_test, p_test, threshold)
rng = np.random.default_rng(42)
n = len(y_test)
boot = {k: [] for k in metric_names}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    mv = metric_values(y_test[idx], p_test[idx], threshold)
    for k in metric_names:
        boot[k].append(mv[k])

rows = []
for k in metric_names:
    arr = np.asarray(boot[k], dtype=float)
    lo = np.nanpercentile(arr, 2.5) if np.isfinite(arr).any() else np.nan
    hi = np.nanpercentile(arr, 97.5) if np.isfinite(arr).any() else np.nan
    rows.append({
        "metric": primary if k == primary else k,
        "value": point[k],
        "ci_low": lo,
        "ci_high": hi,
    })

def r4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

res = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    res[c] = res[c].map(r4)

mlca_runtime.save_table(res, "test_metrics.csv",
                        caption="Test metrics with 95% bootstrap CI (1000 resamples)")
for _, row in res.iterrows():
    mlca_runtime.record_metric(row["metric"], float(row["value"]), "test",
                               ci=(row["ci_low"], row["ci_high"]))
auc_row = res[res["metric"] == "roc_auc"].iloc[0]
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc_row["value"]),
                                   source="test_metrics.csv")

print(f"test rows={len(test)} features={len(feature_cols)}")
print(f"selected={sel_name} positive={positive} threshold={threshold:.4g}")
mrow = res[res["metric"] == primary].iloc[0]
print(f"{primary}={mrow['value']:.4f} CI=[{mrow['ci_low']:.4f},{mrow['ci_high']:.4f}]")
