import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

feature_cols = []
for c in test.columns:
    lc = str(c).lower()
    if c in targets or c in excluded or c == group:
        continue
    if lc in {"id", "row_id", "index"} or lc.endswith("_id") or (lc.endswith("id") and lc != "valid"):
        continue
    feature_cols.append(c)

X_val = val[feature_cols].copy()
X_test = test[feature_cols].copy()
y_val_raw = val[targets[0]]
y_test_raw = test[targets[0]]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_HP31G2ZF/selector.pkl"])

def complete_pipeline(obj):
    if isinstance(obj, Pipeline) and len(obj.steps) > 1:
        return obj
    steps = []
    if isinstance(sel, Pipeline):
        steps.append(("selector", sel))
    else:
        if pre is not None:
            steps.append(("preprocessor", pre))
        if sel is not None:
            steps.append(("selector", sel))
    if isinstance(obj, Pipeline) and len(obj.steps) == 1:
        steps.extend(obj.steps)
    else:
        steps.append(("model", obj))
    return Pipeline(steps)

candidates = []
for name, path in [
    ("logistic", inp["07_model_train_logistic_QH9VENHA/model.pkl"]),
    ("alternative", inp["08_model_train_alternative_2548TA6F/alternative_model.pkl"]),
]:
    candidates.append((name, complete_pipeline(load_pkl(path))))

def find_value(obj, needle):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in str(k).lower() and isinstance(v, (int, float, str)):
                return v
            r = find_value(v, needle)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_value(v, needle)
            if r is not None:
                return r
    return None

pos_label = find_value(proto, "positive")

def classes_of(pipe):
    return list(pipe.named_steps[pipe.steps[-1][0]].classes_)

def proba_pos(pipe, X, pos):
    p = pipe.predict_proba(X)
    cls = classes_of(pipe)
    idx = cls.index(pos) if pos in cls else 1
    return np.asarray(p)[:, idx]

cls0 = classes_of(candidates[0][1])
pos = pos_label if pos_label is not None and pos_label in cls0 else cls0[1]
y_val = np.asarray(y_val_raw == pos, dtype=int)
y_test = np.asarray(y_test_raw == pos, dtype=int)

def score_primary(y, p, threshold=0.5):
    if primary in {"roc_auc", "auc"}:
        return roc_auc_score(y, p)
    if primary in {"average_precision", "pr_auc"}:
        return average_precision_score(y, p)
    if primary == "brier_score":
        return brier_score_loss(y, p)
    pred = (p >= threshold).astype(int)
    if primary == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    if primary in {"f1", "macro_f1"}:
        return recall_score(y, pred, average="macro", zero_division=0)
    return roc_auc_score(y, p)

scores = []
for name, pipe in candidates:
    p = proba_pos(pipe, X_val, pos)
    scores.append((name, pipe, score_primary(y_val, p)))

maximize = primary not in {"log_loss", "brier_score", "rmse", "mae"}
best_name, best_pipe, best_score = (
    max(scores, key=lambda x: x[2]) if maximize else min(scores, key=lambda x: x[2])
)

threshold = None
try:
    vm = pd.read_csv(inp["09_validate_score_validation_GND3XZ5X/validation_metrics.csv"])
    if {"metric", "value"}.issubset(vm.columns):
        m = vm[vm["metric"].astype(str).str.lower().str.contains("threshold")]
        if len(m):
            threshold = float(m.iloc[0]["value"])
except Exception:
    pass

if threshold is None:
    man = json.loads(inp["09_validate_score_validation_GND3XZ5X/manifest.json"].read_text())
    t = find_value(man, "threshold")
    if isinstance(t, (int, float)):
        threshold = float(t)

p_val_best = proba_pos(best_pipe, X_val, pos)
if threshold is None:
    fpr, tpr, ths = roc_curve(y_val, p_val_best)
    threshold = float(ths[np.argmax(tpr - fpr)])

p_test = proba_pos(best_pipe, X_test, pos)

def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier_score": brier_score_loss(y, p),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
    }

metrics = binary_metrics(y_test, p_test, threshold)
names = list(metrics.keys())
boot = {k: [] for k in names}
rng = np.random.default_rng(42)
n = len(y_test)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb = y_test[idx], p_test[idx]
    if len(np.unique(yb)) < 2:
        continue
    try:
        for k, v in binary_metrics(yb, pb, threshold).items():
            boot[k].append(v)
    except Exception:
        continue

rows = []
for k in names:
    arr = boot[k]
    lo, hi = np.nanpercentile(arr, [2.5, 97.5]) if arr else (np.nan, np.nan)
    rows.append({"metric": k, "value": metrics[k], "ci_low": lo, "ci_high": hi})

out_df = pd.DataFrame(rows).round(4)
out_path = mlca_runtime.save_table(out_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

for _, r in out_df.iterrows():
    ci = (float(r["ci_low"]), float(r["ci_high"])) if pd.notna(r["ci_low"]) else None
    mlca_runtime.record_metric(str(r["metric"]), float(r["value"]), split="test", ci=ci)

if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(metrics["roc_auc"]), source=out_path.name)

p_row = out_df[out_df["metric"] == primary].iloc[0]
print(f"features={len(feature_cols)} test_rows={len(test)} selected={best_name}")
print(f"{primary}={p_row['value']:.6f} ci=({p_row['ci_low']:.6f},{p_row['ci_high']:.6f})")
