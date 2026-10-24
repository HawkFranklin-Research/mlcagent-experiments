import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "task" not in proto:
    raise KeyError(f"Missing task. Available top-level keys: {list(proto)}")
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
if isinstance(excluded, dict):
    excluded = excluded.get("columns", []) or []
group = proto.get("group_column")

test = pd.read_parquet(inp["test"])
model = pickle.loads(inp["latest_model"].read_bytes())

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

feats = feature_names(model)
if feats is None:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feats = [c for c in test.columns if c not in drop]
X_test = test.loc[:, feats]
classes = np.asarray(model.classes_)
proba = model.predict_proba(X_test)
y_true = test[targets[0]].to_numpy()
is_binary = task.get("type") == "binary"

def saved_threshold():
    try:
        man = json.loads(inp["09_validate_score_validation_FZA1JB0X/manifest.json"].read_text())
    except Exception:
        man = {}
    def find(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "threshold" in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None
    t = find(man)
    if t is not None:
        return t
    try:
        vm = pd.read_csv(inp["09_validate_score_validation_FZA1JB0X/validation_metrics.csv"])
        if "metric" in vm.columns and "value" in vm.columns:
            m = vm["metric"].astype(str).str.contains("threshold", case=False, na=False)
            if m.any():
                return float(vm.loc[m, "value"].iloc[0])
    except Exception:
        pass
    return None

if is_binary:
    pos_label = task.get("positive_label")
    if pos_label is None:
        pos_label = classes[1]
    pos_idx = int(np.where(classes == pos_label)[0][0])
    neg_label = classes[0] if pos_label == classes[1] else classes[1]
    proba_pos = proba[:, pos_idx]
    threshold = saved_threshold()
    if threshold is None:
        val = pd.read_parquet(inp["val"])
        pv = model.predict_proba(val.loc[:, feats])[:, pos_idx]
        yv = (val[targets[0]].to_numpy() == pos_label).astype(int)
        best_t, best_s = 0.5, -1.0
        for t in np.unique(np.r_[0.0, pv, 1.0]):
            s = balanced_accuracy_score(yv, (pv >= t).astype(int))
            if s > best_s:
                best_t, best_s = float(t), s
        threshold = best_t
    y_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (proba_pos >= threshold).astype(int)
    y_pred = np.where(y_pred_bin == 1, pos_label, neg_label)

    def calc(yb, pp, ypb):
        out = {}
        out["roc_auc"] = roc_auc_score(yb, pp) if len(np.unique(yb)) == 2 else np.nan
        out["pr_auc"] = average_precision_score(yb, pp) if yb.sum() > 0 else np.nan
        out["balanced_accuracy"] = balanced_accuracy_score(yb, ypb) if len(np.unique(yb)) == 2 else np.nan
        out["brier_score"] = brier_score_loss(yb, pp)
        tp = int(((yb == 1) & (ypb == 1)).sum()); tn = int(((yb == 0) & (ypb == 0)).sum())
        fp = int(((yb == 0) & (ypb == 1)).sum()); fn = int(((yb == 1) & (ypb == 0)).sum())
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
        return out

    point = calc(y_bin, proba_pos, y_pred_bin)
    n = len(y_bin)
    rng = np.random.default_rng(42)
    boot = {k: [] for k in point}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            r = calc(y_bin[idx], proba_pos[idx], y_pred_bin[idx])
            for k, v in r.items():
                if np.isfinite(v):
                    boot[k].append(v)
        except Exception:
            pass
else:
    y_pred = model.predict(X_test)
    missing = [c for c in classes if not (y_true == c).any()]
    if missing:
        print(f"classes absent from test: {missing}")

    def calc(yt, pp, yp):
        vals = []
        for j, c in enumerate(classes):
            yc = (yt == c).astype(int)
            if len(np.unique(yc)) == 2:
                vals.append(roc_auc_score(yc, pp[:, j]))
        return {
            "roc_auc_ovr_macro": float(np.mean(vals)) if vals else np.nan,
            "f1_macro": f1_score(yt, yp, labels=classes, average="macro", zero_division=0),
        }

    point = calc(y_true, proba, y_pred)
    n = len(y_true)
    rng = np.random.default_rng(42)
    boot = {k: [] for k in point}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            r = calc(y_true[idx], proba[idx], y_pred[idx])
            for k, v in r.items():
                if np.isfinite(v):
                    boot[k].append(v)
        except Exception:
            pass

rows = []
for k, v in point.items():
    vals = boot[k]
    lo = float(np.percentile(vals, 2.5)) if len(vals) >= 2 else np.nan
    hi = float(np.percentile(vals, 97.5)) if len(vals) >= 2 else np.nan
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows)
metrics_df = metrics_df.sort_values("metric", key=lambda s: s != primary)

primary_row = metrics_df.loc[metrics_df["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(primary_row["value"]), "test",
                           ci=(float(primary_row["ci_low"]), float(primary_row["ci_high"])))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(primary_row["value"]), source="test_metrics.csv")

def sig(x):
    return float(f"{x:.4g}") if pd.notna(x) and np.isfinite(x) else x
metrics_df[["value", "ci_low", "ci_high"]] = metrics_df[["value", "ci_low", "ci_high"]].map(sig)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pred = pd.DataFrame({"row_id": test.index, "partition": "test"})
if len(targets) == 1:
    pred["y_true"] = y_true
    pred["y_pred"] = y_pred
else:
    for t in targets:
        pred[f"y_true__{t}"] = test[t].to_numpy()
        pred[f"y_pred__{t}"] = y_pred
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print(f"test rows={len(test)} features={X_test.shape[1]} classes={classes.tolist()}")
if is_binary:
    print(f"threshold={threshold:.4f}")
print(f"{primary}={primary_row['value']:.4f} ci=({primary_row['ci_low']:.4f},{primary_row['ci_high']:.4f})")
