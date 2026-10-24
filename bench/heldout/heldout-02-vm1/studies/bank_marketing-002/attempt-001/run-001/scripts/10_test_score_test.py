import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
target = targets[0]

test_df = pd.read_parquet(inp["test"])
with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)
classes = np.asarray(model.classes_)
try:
    feat_cols = list(model.feature_names_in_)
except Exception:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feat_cols = [c for c in test_df.columns if c not in drop]

X = test_df[feat_cols].copy()
y_true = test_df[target].values
proba = model.predict_proba(X)
n = len(y_true)

def find_thr(d, depth=0):
    if depth > 4 or not isinstance(d, dict): return None
    for k, v in d.items():
        if "thresh" in k.lower():
            try: return float(v)
            except Exception: pass
        r = find_thr(v, depth + 1)
        if r is not None: return r
    return None

threshold = find_thr(task)
if threshold is None:
    try:
        vmani = json.loads(inp["09_validate_score_validation_C91Z2840/manifest.json"].read_text())
        threshold = find_thr(vmani)
    except Exception: pass
if threshold is None:
    threshold = 0.5
print(f"threshold={threshold}")

rng = np.random.default_rng(42)
boot_idx = [rng.integers(0, n, n) for _ in range(1000)]

def ci_of(fval):
    vals = []
    for i in boot_idx:
        try: vals.append(fval(i))
        except Exception: pass
    if not vals: return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

results = []
def add(name, fval):
    v = float(fval(np.arange(n)))
    lo, hi = ci_of(fval)
    results.append({"metric": name, "value": v, "ci_low": lo, "ci_high": hi})

binary = (task.get("type") == "binary") or (len(classes) == 2)
if binary:
    pos = task.get("positive_label") or classes[1]
    pi = int(np.where(classes == pos)[0][0])
    p = proba[:, pi]
    yb = (y_true == pos).astype(int)
    yp = (p >= threshold).astype(int)
    add("roc_auc", lambda i: roc_auc_score(yb[i], p[i]))
    add("pr_auc", lambda i: average_precision_score(yb[i], p[i]))
    add("balanced_accuracy", lambda i: balanced_accuracy_score(yb[i], yp[i]))
    add("brier_score", lambda i: brier_score_loss(yb[i], p[i]))
    def sens(i):
        cm = confusion_matrix(yb[i], yp[i], labels=[0, 1])
        d = cm[1, 0] + cm[1, 1]
        return cm[1, 1] / d if d else float('nan')
    def spec(i):
        cm = confusion_matrix(yb[i], yp[i], labels=[0, 1])
        d = cm[0, 0] + cm[0, 1]
        return cm[0, 0] / d if d else float('nan')
    add("sensitivity", sens)
    add("specificity", spec)
    y_pred_label = np.where(yp == 1, pos, classes[1 - pi])
else:
    y_pred_label = classes[proba.argmax(axis=1)]
    present = [c for c in classes if c in set(y_true)]
    if len(present) < len(classes):
        print("classes excluded from macro AUC:", [c for c in classes if c not in present])
        def m_auc(i):
            ys = y_true[i]
            pre = set(ys)
            aucs = [roc_auc_score((ys == c).astype(int), proba[i, j])
                    for j, c in enumerate(classes) if c in pre]
            return float(np.mean(aucs))
    else:
        def m_auc(i):
            return roc_auc_score(y_true[i], proba[i], multi_class='ovr', average='macro', labels=classes)
    add("roc_auc", m_auc)
    add("f1_macro", lambda i: f1_score(y_true[i], y_pred_label[i], average='macro'))

df = pd.DataFrame(results).round(4)
mlca_runtime.save_table(df, "test_metrics")

prim_row = df[df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prim_row["value"]), "test",
                           ci=(float(prim_row["ci_low"]), float(prim_row["ci_high"])))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(prim_row["value"]),
                                       source="test_metrics.csv")

pred_df = pd.DataFrame({"row_id": test_df.index, "partition": "test"})
pred_df[f"y_true__{target}"] = y_true
pred_df[f"y_pred__{target}"] = y_pred_label
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

print("test rows:", n, "| metrics rows:", len(df))
