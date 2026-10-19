import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, f1_score)

mlca_runtime.set_seed(42)
SEED = 42
rng = np.random.default_rng(SEED)
inp = mlca_runtime.inputs()

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def input_names(o):
    if hasattr(o, "feature_names_in_"):
        return list(o.feature_names_in_)
    if isinstance(o, Pipeline) and o.steps:
        return input_names(o.steps[0][1])
    return None

def find_key(obj, needles):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(n in str(k).lower() for n in needles):
                return v
            r = find_key(v, needles)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, needles)
            if r is not None:
                return r
    return None

def resolve_input(val):
    s = str(val)
    for k, p in inp.items():
        if s == k or s == p.name or s.endswith(k) or k.endswith(Path(s).name):
            return p
    return None

def build_predictor(model_path, raw_set):
    model = load_pickle(model_path)
    mn = input_names(model)
    if mn and set(mn).issubset(raw_set):
        return model, mn
    pre = load_pickle(inp["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"])
    sel = load_pickle(inp["06_features_select_features_PDWSHJGW/selector.pkl"])
    sn = input_names(sel)
    if sn and set(sn).issubset(raw_set):
        return Pipeline([("selector", sel), ("model", model)]), sn
    pn = input_names(pre)
    if pn and set(pn).issubset(raw_set):
        steps = [("preprocessor", pre)]
        if sn is not None:
            steps.append(("selector", sel))
        steps.append(("model", model))
        return Pipeline(steps), pn
    return model, list(raw_set)

protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing _protocol['{k}']; available keys: {list(protocol.keys())}")
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])
raw_cols = [c for c in test.columns if c not in set(targets + excluded + ([group_col] if group_col else []))]
for c in list(raw_cols):
    if c in train:
        s = train[c]
        key_like = (s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s))
        if key_like and len(s) > 1 and s.nunique(dropna=False) == len(s):
            raw_cols.remove(c)
raw_set = set(raw_cols)

man9 = json.loads(inp["09_validate_score_validation_X45306SJ/manifest.json"].read_text())
selected = find_key(man9, ["selected_model", "best_model", "chosen_model", "model_path", "selected_model_name"])
selected_path = resolve_input(selected) if selected else None
if selected_path is None:
    selected_path = inp["latest_model"]

thr = find_key(man9, ["threshold", "cutoff", "decision_threshold"])
for mk in ["07_model_train_logistic_H2ZAD1XK/manifest.json", "08_model_train_alternative_XPWMPVN8/manifest.json"]:
    if thr is None and mk in inp:
        thr = find_key(json.loads(inp[mk].read_text()), ["threshold", "cutoff", "decision_threshold"])
try:
    threshold = float(thr)
    if not 0 <= threshold <= 1:
        threshold = 0.5
except Exception:
    threshold = 0.5

predictor, names = build_predictor(selected_path, raw_set)
X_test = test[names]
proba = predictor.predict_proba(X_test)
classes = list(predictor.classes_)
y_raw = test[targets[0]].to_numpy()

def boot_ci(fn, n):
    vals = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
            if np.isfinite(v):
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

if len(classes) == 2:
    pos = task.get("positive_label", classes[1])
    if pos not in classes:
        pos = classes[1]
    j = classes.index(pos)
    yb = (y_raw == pos).astype(int)
    yp = (proba[:, j] >= threshold).astype(int)
    metric_fns = {
        "roc_auc": lambda idx: roc_auc_score(yb[idx], proba[idx, j]),
        "pr_auc": lambda idx: average_precision_score(yb[idx], proba[idx, j]),
        "balanced_accuracy": lambda idx: balanced_accuracy_score(yb[idx], yp[idx]),
        "brier_score": lambda idx: brier_score_loss(yb[idx], proba[idx, j]),
        "sensitivity": lambda idx: recall_score(yb[idx], yp[idx], pos_label=1, zero_division=0),
        "specificity": lambda idx: recall_score(yb[idx], yp[idx], pos_label=0, zero_division=0),
    }
    if primary not in metric_fns:
        metric_fns[primary] = metric_fns["roc_auc"]
else:
    yp = predictor.predict(X_test)
    def macro_ovr(idx):
        vals = []
        for c in np.unique(y_raw[idx]):
            if c in classes:
                cc = (y_raw[idx] == c).astype(int)
                if len(np.unique(cc)) > 1:
                    vals.append(roc_auc_score(cc, proba[idx, classes.index(c)]))
        return float(np.mean(vals)) if vals else np.nan
    metric_fns = {
        "roc_auc": macro_ovr,
        "macro_f1": lambda idx: f1_score(y_raw[idx], yp[idx], average="macro", labels=classes),
    }
    if primary not in metric_fns:
        metric_fns[primary] = metric_fns["macro_f1"] if "f1" in primary.lower() else macro_ovr

rows = []
for name, fn in metric_fns.items():
    value = float(fn(np.arange(len(y_raw))))
    lo, hi = boot_ci(fn, len(y_raw))
    rows.append({"metric": name, "value": value, "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows, columns=["metric", "value", "ci_low", "ci_high"])
for c in ["value", "ci_low", "ci_high"]:
    metrics_df[c] = metrics_df[c].map(lambda x: float(f"{x:.4g}") if np.isfinite(x) else np.nan)

mlca_runtime.save_table(metrics_df, "test_metrics.csv", "Frozen model test metrics with 95% bootstrap CI")
prow = metrics_df.loc[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prow["value"]), "test",
                           ci=(float(prow["ci_low"]), float(prow["ci_high"])))
ra = metrics_df.loc[metrics_df["metric"] == "roc_auc"]
if len(ra):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(ra.iloc[0]["value"]), source="test_metrics.csv")

print(f"test rows={len(test)} features={len(names)}")
print(f"model={Path(selected_path).name} threshold={threshold:.4g}")
print(f"{primary}={float(prow['value']):.6g} CI=({float(prow['ci_low']):.6g},{float(prow['ci_high']):.6g})")
