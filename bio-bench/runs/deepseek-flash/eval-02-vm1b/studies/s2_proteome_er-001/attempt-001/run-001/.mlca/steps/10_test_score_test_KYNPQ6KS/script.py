import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}")
primary = task["primary_metric"]
target = list(targets)[0] if isinstance(targets, (list, tuple)) else targets
pos_decl = task.get("positive_label", protocol.get("positive_label"))

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

models = {
    "logistic": load_pkl("07_model_train_logistic_HMHZA6MZ/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_R3YZHT6E/alternative_model.pkl"),
}

def feature_names(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise AttributeError("Saved model does not expose feature_names_in_")

val = pd.read_parquet(inp["val"])
def val_auc(m):
    feats = feature_names(m)
    classes = list(m.classes_)
    pos = pos_decl if pos_decl in classes else classes[1]
    j = classes.index(pos)
    y = (val[target].to_numpy() == pos).astype(int)
    return roc_auc_score(y, m.predict_proba(val.loc[:, feats])[:, j])

scores = {k: val_auc(m) for k, m in models.items()}
choice = max(scores, key=scores.get)
model = models[choice]

test = pd.read_parquet(inp["test"])
feats = feature_names(model)
classes = list(model.classes_)
if len(classes) != 2:
    raise ValueError("Binary task expected")
pos = pos_decl if pos_decl in classes else classes[1]
j = classes.index(pos)
neg = classes[1 - j]
proba = model.predict_proba(test.loc[:, feats])
proba_pos = proba[:, j]
y_true = test[target].to_numpy()
y_bin = (y_true == pos).astype(int)

val_metrics = pd.read_csv(inp["09_validate_score_validation_VVC5CYMZ/validation_metrics.csv"])
val_manifest = json.loads(inp["09_validate_score_validation_VVC5CYMZ/manifest.json"].read_text())
def find_threshold(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float, np.number)):
                return float(v)
            r = find_threshold(v)
            if r is not None:
                return r
    elif isinstance(x, list):
        for v in x:
            r = find_threshold(v)
            if r is not None:
                return r
    return None
threshold = find_threshold(val_manifest)
if threshold is None:
    for c in val_metrics.columns:
        if "threshold" in c.lower():
            s = pd.to_numeric(val_metrics[c], errors="coerce").dropna()
            if len(s):
                threshold = float(s.iloc[0]); break
if threshold is None:
    cols = {c.lower(): c for c in val_metrics.columns}
    mc, vc = cols.get("metric"), cols.get("value")
    if mc and vc:
        sub = val_metrics[val_metrics[mc].astype(str).str.lower().str.contains("threshold")]
        s = pd.to_numeric(sub[vc], errors="coerce").dropna() if len(sub) else pd.Series(dtype=float)
        if len(s):
            threshold = float(s.iloc[0])
if threshold is None:
    threshold = 0.5

def metric_values(yb, pp, thr):
    pred = (pp >= thr).astype(int)
    out = {}
    for name, fn in [
        ("roc_auc", lambda: roc_auc_score(yb, pp)),
        ("pr_auc", lambda: average_precision_score(yb, pp)),
        ("balanced_accuracy", lambda: balanced_accuracy_score(yb, pred)),
        ("brier_score", lambda: brier_score_loss(yb, pp)),
    ]:
        try:
            out[name] = float(fn())
        except Exception:
            out[name] = np.nan
    tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out

metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
if primary not in metric_names:
    raise ValueError(f"Unsupported primary_metric {primary}")
base = metric_values(y_bin, proba_pos, threshold)
rng = np.random.default_rng(42)
boot = {k: [] for k in metric_names}
n = len(y_bin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    vals = metric_values(y_bin[idx], proba_pos[idx], threshold)
    for k in metric_names:
        if not np.isnan(vals[k]):
            boot[k].append(vals[k])

rows = []
for k in metric_names:
    arr = np.asarray(boot[k], float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
    rows.append({"metric": k, "value": base[k], "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows)
out_df = metrics_df.copy()
for c in ["value", "ci_low", "ci_high"]:
    out_df[c] = out_df[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

row = metrics_df[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(row["value"]), "test", ci=(float(row["ci_low"]), float(row["ci_high"])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(base["roc_auc"]), source="test_metrics.csv")
mlca_runtime.save_table(out_df, "test_metrics.csv", "Test metrics with 95% bootstrap CI")

pred_labels = np.where(proba_pos >= threshold, pos, neg)
pred_df = pd.DataFrame({"row_id": test.index, "partition": "test"})
if len(list(targets)) == 1:
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = pred_labels
else:
    pred_df[f"y_true__{target}"] = y_true
    pred_df[f"y_pred__{target}"] = pred_labels
for c, col in zip(classes, proba.T):
    pred_df[f"proba__{c}"] = col
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

print(f"test rows={len(test)} features={len(feats)}")
print(f"selected={choice} threshold={threshold:.4f}")
print(f"{primary}={base[primary]:.4f} CI=({row['ci_low']:.4f},{row['ci_high']:.4f})")
