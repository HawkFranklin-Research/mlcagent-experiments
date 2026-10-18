import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score, f1_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"] if isinstance(protocol["targets"], list) else [protocol["targets"]]
target = targets[0]
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
test = pd.read_parquet(inp["test"])

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = load(inp["latest_model"])
if isinstance(model, dict):
    model = next((model[k] for k in ("model", "pipeline", "estimator") if k in model), model)
if not hasattr(model, "predict_proba"):
    pre = load(inp["05_preprocess_fit_preprocessing_599QM3CC/preprocessor.pkl"])
    sel = load(inp["06_features_select_features_Y69A7R1X/selector.pkl"])
    if isinstance(sel, Pipeline) and "preprocessor" in sel.named_steps:
        model = Pipeline([("selector", sel), ("model", model)])
    else:
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

def classes_of(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise ValueError("no classes_")

def features_of(m):
    if hasattr(m, "feature_names_in_") and m.feature_names_in_ is not None:
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_") and step.feature_names_in_ is not None:
                return list(step.feature_names_in_)
    return None

features = features_of(model)
if features is None:
    protected = set(targets) | excluded | ({group} if group else set())
    features = []
    for c in test.columns:
        if c in protected:
            continue
        s = test[c]
        if s.dtype.kind in "OUS" and s.notna().all() and s.nunique() == len(s):
            continue
        features.append(c)
X = test[features].copy()
classes = classes_of(model)
proba = model.predict_proba(X)

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_thr(v)
            if r is not None:
                return r
    return None

thr = None
for k in sorted([k for k in inp if k.endswith("manifest.json")], key=lambda x: 0 if "09_validate" in x else 1):
    try:
        thr = find_thr(json.loads(inp[k].read_text()))
    except Exception:
        pass
    if thr is not None:
        break
if thr is None:
    thr = find_thr(protocol)
if thr is None:
    thr = 0.5

y = test[target].values
if len(classes) == 2:
    pos = task.get("positive_label", protocol.get("positive_label", classes[1]))
    if pos not in classes:
        pos = classes[1]
    p = proba[:, list(classes).index(pos)]
    ybin = (y == pos).astype(int)
    if ybin.sum() == 0:
        ybin = (pd.Series(y).astype(str).str.strip().str.rstrip(".") == str(pos).strip().rstrip(".")).astype(int).values

    def calc():
        pred = (p >= thr).astype(int)
        out = {}
        for name, fn in [
            ("roc_auc", lambda: roc_auc_score(ybin, p)),
            ("pr_auc", lambda: average_precision_score(ybin, p)),
            ("balanced_accuracy", lambda: balanced_accuracy_score(ybin, pred)),
            ("brier_score", lambda: brier_score_loss(ybin, p)),
            ("sensitivity", lambda: recall_score(ybin, pred, zero_division=0)),
            ("specificity", lambda: recall_score(1 - ybin, 1 - pred, zero_division=0)),
        ]:
            try:
                out[name] = float(fn())
            except Exception:
                out[name] = np.nan
        return out
else:
    ytrue = y

    def calc():
        out = {}
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(ytrue, proba, multi_class="ovr", average="macro", labels=classes))
        except Exception:
            out["roc_auc_ovr"] = np.nan
        try:
            out["f1_macro"] = float(f1_score(ytrue, classes[np.argmax(proba, axis=1)], average="macro", zero_division=0))
        except Exception:
            out["f1_macro"] = np.nan
        return out

base = calc()
rng = np.random.default_rng(42)
boot = {k: [] for k in base}
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    if len(classes) == 2:
        yb, pb = ybin[idx], p[idx]
        pred = (pb >= thr).astype(int)
        for name, fn in [
            ("roc_auc", lambda: roc_auc_score(yb, pb)),
            ("pr_auc", lambda: average_precision_score(yb, pb)),
            ("balanced_accuracy", lambda: balanced_accuracy_score(yb, pred)),
            ("brier_score", lambda: brier_score_loss(yb, pb)),
            ("sensitivity", lambda: recall_score(yb, pred, zero_division=0)),
            ("specificity", lambda: recall_score(1 - yb, 1 - pred, zero_division=0)),
        ]:
            try:
                boot[name].append(float(fn()))
            except Exception:
                boot[name].append(np.nan)
    else:
        yb, prob = ytrue[idx], proba[idx]
        try:
            boot["roc_auc_ovr"].append(float(roc_auc_score(yb, prob, multi_class="ovr", average="macro", labels=classes)))
        except Exception:
            boot["roc_auc_ovr"].append(np.nan)
        try:
            boot["f1_macro"].append(float(f1_score(yb, classes[np.argmax(prob, axis=1)], average="macro", zero_division=0)))
        except Exception:
            boot["f1_macro"].append(np.nan)

rows = []
for k, v in base.items():
    vals = np.asarray(boot[k], dtype=float)
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
df = pd.DataFrame(rows, columns=["metric", "value", "ci_low", "ci_high"]).round(6)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

row = df[df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(row["value"]), "test", ci=(float(row["ci_low"]), float(row["ci_high"])))
auc_row = df[df["metric"] == "roc_auc"]
if len(auc_row):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(auc_row.iloc[0]["value"]), source="test_metrics.csv")

print(f"test rows={len(test)} features={len(features)}")
print(f"primary {primary}={row['value']:.4f} ci=({row['ci_low']:.4f},{row['ci_high']:.4f})")
print(f"classes={list(classes)} threshold={thr}")
