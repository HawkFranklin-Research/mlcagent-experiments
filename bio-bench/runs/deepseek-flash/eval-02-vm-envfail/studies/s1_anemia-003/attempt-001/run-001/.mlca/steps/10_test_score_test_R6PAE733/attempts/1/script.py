import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score,
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "task" not in proto:
    raise KeyError(f"Missing 'task'. Available top-level keys: {list(proto)}")
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric. Available task keys: {list(task)}")
if "targets" not in proto:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {list(proto)}")

primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
target = targets[0]
test = pd.read_parquet(inp["test"])
if target not in test.columns:
    raise KeyError(f"Target {target!r} not found in test columns")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_number(obj, needle):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in k.lower():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        pass
            r = find_number(v, needle)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_number(v, needle)
            if r is not None:
                return r
    return None

def find_selected(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if ("selected" in k.lower() or "best" in k.lower()) and isinstance(v, str):
                return v
            r = find_selected(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_selected(v)
            if r:
                return r
    return None

manifest_keys = [
    "09_validate_score_validation_F0FWKXE9/manifest.json",
    "07_model_train_logistic_21TNHNVT/manifest.json",
    "08_model_train_alternative_67TQZBPK/manifest.json",
]
threshold = None
selected = None
for key in manifest_keys:
    manifest = json.loads(inp[key].read_text())
    if threshold is None:
        threshold = find_number(manifest, "threshold")
    if selected is None:
        selected = find_selected(manifest)

for key in [
    "09_validate_score_validation_F0FWKXE9/validation_metrics.parquet",
    "09_validate_score_validation_F0FWKXE9/validation_metrics.csv",
]:
    if threshold is not None:
        break
    try:
        vm = pd.read_parquet(inp[key]) if key.endswith(".parquet") else pd.read_csv(inp[key])
    except Exception:
        continue
    for c in vm.columns:
        if "threshold" in c.lower():
            s = pd.to_numeric(vm[c], errors="coerce").dropna()
            if len(s):
                threshold = float(s.iloc[0])
                break
    if threshold is None and "metric" in vm.columns and "value" in vm.columns:
        s = vm[vm["metric"].astype(str).str.contains("threshold", case=False, na=False)]
        if len(s):
            threshold = float(s["value"].iloc[0])

main_model = load_pkl(inp["07_model_train_logistic_21TNHNVT/model.pkl"])
alt_model = load_pkl(inp["08_model_train_alternative_67TQZBPK/alternative_model.pkl"])
if selected and ("alt" in selected.lower() or "08" in selected):
    chosen = alt_model
elif selected:
    chosen = main_model
else:
    chosen = main_model
    try:
        vm = pd.read_parquet(inp["09_validate_score_validation_F0FWKXE9/validation_metrics.parquet"])
        mcol = next((c for c in vm.columns if "model" in c.lower() or "candidate" in c.lower()), None)
        if mcol and "metric" in vm.columns and "value" in vm.columns:
            sub = vm[vm["metric"].astype(str).str.contains(primary, case=False, na=False)]
            if len(sub):
                agg = sub.groupby(mcol)["value"].mean()
                best = agg.idxmax() if primary in {"roc_auc", "pr_auc", "balanced_accuracy", "f1", "accuracy"} else agg.idxmin()
                chosen = alt_model if ("alt" in str(best).lower() or "08" in str(best)) else main_model
    except Exception:
        pass

if not isinstance(chosen, Pipeline):
    raise TypeError("Selected saved model is not a Pipeline")

feat_cols = list(getattr(chosen, "feature_names_in_", []))
if not feat_cols:
    feat_cols = [c for c in test.columns if c not in set(targets + excluded) and c != group]
X = test[feat_cols]
y = test[target].values
proba = chosen.predict_proba(X)
classes = chosen.classes_
pos = task.get("positive_label", task.get("positive_class", classes[1]))
pos_idx = list(classes).index(pos)
p = proba[:, pos_idx]

if threshold is None:
    val = pd.read_parquet(inp["val"])
    pv = chosen.predict_proba(val[feat_cols])[:, pos_idx]
    yv = (val[target].values == pos).astype(int)
    candidates = np.unique(pv)
    bas = [balanced_accuracy_score(yv, (pv >= t).astype(int)) for t in candidates]
    threshold = float(candidates[int(np.argmax(bas))])

neg_idx = 1 - pos_idx
y_pred = np.where(p >= threshold, classes[pos_idx], classes[neg_idx])

def calc_metrics(yy, pp, yp):
    yb = (yy == pos).astype(int)
    ypb = (yp == pos).astype(int)
    return {
        "roc_auc": roc_auc_score(yb, pp),
        "pr_auc": average_precision_score(yb, pp),
        "balanced_accuracy": balanced_accuracy_score(yb, ypb),
        "brier_score": brier_score_loss(yb, pp),
        "sensitivity": recall_score(yb, ypb, pos_label=1, zero_division=0),
        "specificity": recall_score(yb, ypb, pos_label=0, zero_division=0),
    }

metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
main = calc_metrics(y, p, y_pred)
rng = np.random.default_rng(42)
boot = {k: [] for k in metric_names}
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    try:
        m = calc_metrics(y[idx], p[idx], y_pred[idx])
        for k in metric_names:
            boot[k].append(m[k])
    except Exception:
        pass

rows = []
for k in metric_names:
    lo = float(np.nanpercentile(boot[k], 2.5)) if boot[k] else np.nan
    hi = float(np.nanpercentile(boot[k], 97.5)) if boot[k] else np.nan
    rows.append({"metric": k, "value": main[k], "ci_low": lo, "ci_high": hi})
res = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    res[c] = res[c].round(4)
mlca_runtime.save_table(res, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
for _, r in res.iterrows():
    mlca_runtime.record_metric(r["metric"], float(r["value"]), "test", (float(r["ci_low"]), float(r["ci_high"])))
if primary == "roc_auc":
    v = float(res.loc[res["metric"] == primary, "value"].iloc[0])
    mlca_runtime.record_fact_candidate("test_roc_auc", v, source="test_metrics.csv")

pred = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y, "y_pred": y_pred})
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print(f"test rows: {len(test)}")
print(f"{primary}: {main[primary]:.4f} CI [{res.loc[res.metric == primary, 'ci_low'].iloc[0]:.4f}, {res.loc[res.metric == primary, 'ci_high'].iloc[0]:.4f}]")
print(f"threshold: {threshold:.4f}")
print(f"metrics: {len(res)}")
