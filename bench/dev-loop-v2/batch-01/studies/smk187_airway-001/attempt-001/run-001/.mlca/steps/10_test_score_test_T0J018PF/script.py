import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary_metric = task["primary_metric"]
task_type = task["type"]
positive_label = task.get("positive_label")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

with open(inputs["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_NTK41RPZ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def frontend(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for s in obj.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

# Determine feature columns from the model's expected input names
raw_feats = frontend(model) or frontend(sel) or frontend(pre)
if raw_feats is None:
    feat_cols = [c for c in test.columns if c not in targets and c not in excluded and c != group_col]
else:
    feat_cols = raw_feats

X_test_raw = test[feat_cols]
y_true_all = test[targets].copy()

# Apply preprocessing/selection as needed - use the most complete pipeline
def predict_proba(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)
    return None

# Determine if model already includes preprocessor
def steps_of(obj):
    if hasattr(obj, "named_steps"):
        return [type(v).__name__ for v in obj.named_steps.values()]
    return [type(obj).__name__]

# Build the chain: pre -> sel -> model, skipping pieces already inside later steps
chain = []
model_has_proba = hasattr(model, "predict_proba")
model_inputs = frontend(model)

# Simplest: use model directly on raw if it accepts raw; else apply pre/selector
if model_inputs is not None and set(model_inputs) <= set(test.columns):
    # model expects columns already present (possibly transformed name cols won't match)
    try:
        proba = model.predict_proba(X_test_raw)
    except Exception:
        proba = None
else:
    proba = None

if proba is None:
    Xt = X_test_raw
    for step in (pre, sel, model):
        if step is None:
            continue
        try:
            needs = frontend(step)
            if needs is not None and not all(c in Xt.columns for c in needs):
                # apply as transformer producing new frame or array
                Xt = step.transform(Xt)
            elif hasattr(step, "predict_proba"):
                proba = step.predict_proba(Xt)
            else:
                Xt = step.transform(Xt)
        except Exception as e:
            raise RuntimeError(f"chain failed at {type(step).__name__}: {e}")
    if proba is None:
        proba = model.predict_proba(Xt)

classes = list(model.classes_)
n = len(test)
pred = model.predict(X_test_raw if model_inputs is not None and all(c in test.columns for c in model_inputs) else (Xt if 'Xt' in dir() else X_test_raw))

# Build row outputs
row_df = pd.DataFrame({"row_id": test.index, "partition": "test"}, index=test.index)
for t in targets:
    row_df[f"y_true__{t}"] = y_true_all[t].values
    row_df[f"y_pred__{t}"] = pred
for j, c in enumerate(classes):
    row_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(row_df, "test_predictions.parquet")

# Metrics
y = y_true_all[targets[0]].values
def boot_ci(fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

metrics = {}
ci_map = {}

if task_type == "binary":
    pos = positive_label if positive_label is not None else classes[1]
    pos_idx = classes.index(pos)
    p_pos = proba[:, pos_idx]
    y_bin = (y == pos).astype(int)
    defs = {
        "roc_auc": lambda: roc_auc_score(y_bin, p_pos),
        "pr_auc": lambda: average_precision_score(y_bin, p_pos),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_bin, (p_pos >= 0.5).astype(int)),
        "brier": lambda: brier_score_loss(y_bin, p_pos),
    }
    for name, fn in defs.items():
        metrics[name] = float(fn())
        ci_map[name] = boot_ci(lambda idx: fn())
    # threshold-based sens/spec using 0.5
    yhat = (p_pos >= 0.5).astype(int)
    tp = ((yhat == 1) & (y_bin == 1)).sum(); fn_ = ((yhat == 0) & (y_bin == 1)).sum()
    tn = ((yhat == 0) & (y_bin == 0)).sum(); fp = ((yhat == 1) & (y_bin == 0)).sum()
    sens = tp / (tp + fn_) if (tp + fn_) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    metrics["sensitivity"] = float(sens)
    metrics["specificity"] = float(spec)
    ci_map["sensitivity"] = (np.nan, np.nan)
    ci_map["specificity"] = (np.nan, np.nan)
else:
    present = [c for c in classes if c in np.unique(y)]
    def macro_auc(idx):
        vals = []
        for c in present:
            yc = (y[idx] == c).astype(int)
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            vals.append(roc_auc_score(yc, proba[idx, classes.index(c)]))
        return float(np.mean(vals))
    metrics["roc_auc_ovr"] = macro_auc(np.arange(n))
    ci_map["roc_auc_ovr"] = boot_ci(macro_auc)
    metrics["macro_f1"] = float(f1_score(y, pred, average="macro"))
    ci_map["macro_f1"] = boot_ci(lambda idx: f1_score(y[idx], pred[idx], average="macro"))

rows = []
for name, v in metrics.items():
    lo, hi = ci_map.get(name, (np.nan, np.nan))
    rows.append({"metric": name, "value": v, "ci_low": lo, "ci_high": hi})
mt = pd.DataFrame(rows)
mlca_runtime.save_table(mt, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pv = metrics.get(primary_metric)
cl, ch = ci_map.get(primary_metric, (np.nan, np.nan))
mlca_runtime.record_metric(primary_metric, float(pv), "test", ci=[float(cl), float(ch)])
if primary_metric == "roc_auc" and "roc_auc" in metrics:
    mlca_runtime.record_fact_candidate("test_roc_auc", float(metrics["roc_auc"]),
                                      source="test_metrics.csv")
print(f"n_test={n} classes={classes}")
print(f"{primary_metric}={pv:.4f} [{cl:.4f},{ch:.4f}]")
