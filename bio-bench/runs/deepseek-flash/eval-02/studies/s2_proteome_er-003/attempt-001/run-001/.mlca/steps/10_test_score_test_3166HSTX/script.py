import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score

mlca_runtime.set_seed(42)
rng = np.random.default_rng(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]
task = protocol["task"]
metric_name = task["primary_metric"]
task_type = task.get("type", "binary")
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
test = pd.read_parquet(inputs["test"])

vm = pd.read_csv(inputs["09_validate_score_validation_GCJQ51QV/validation_metrics.csv"])
check = {"metric", "value", "model"}
sel_path = None
if check.issubset(vm.columns):
    rows = vm[vm["metric"].astype(str).str.lower() == metric_name.lower()].copy()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=["value"])
    if not rows.empty:
        loss = {"brier_score", "log_loss", "mae", "rmse", "mse"}
        best = rows.sort_values("value", ascending=metric_name.lower() in loss).iloc[0]
        mname = str(best["model"]).lower()
        if "alt" in mname or "08" in mname:
            sel_path = inputs["08_model_train_alternative_Z005D6HS/alternative_model.pkl"]
        elif "log" in mname or "07" in mname:
            sel_path = inputs["07_model_train_logistic_8VZRZDX0/model.pkl"]
if sel_path is None:
    sel_path = inputs["latest_model"]

def deep_find(obj, needle):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in str(k).lower():
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
            found = deep_find(v, needle)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = deep_find(v, needle)
            if found is not None:
                return found
    return None

threshold = None
if {"metric", "value"}.issubset(vm.columns):
    tr = vm[vm["metric"].astype(str).str.contains("thresh", case=False, na=False)]
    if not tr.empty:
        threshold = float(pd.to_numeric(tr["value"], errors="coerce").dropna().iloc[0])
if threshold is None:
    vmanifest = json.loads(inputs["09_validate_score_validation_GCJQ51QV/manifest.json"].read_text())
    threshold = deep_find(vmanifest, "thresh")
if threshold is None:
    threshold = 0.5

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

base = load_pickle(sel_path)
if isinstance(base, Pipeline):
    predictor = base
else:
    pre = load_pickle(inputs["05_preprocess_fit_preprocessing_B1ZJVKDM/preprocessor.pkl"])
    sel = load_pickle(inputs["06_features_select_features_X6VS56A4/selector.pkl"])
    sel_is_full = isinstance(sel, Pipeline) and len(sel.named_steps) > 1
    if sel_is_full and getattr(sel, "feature_names_in_", None) is not None:
        predictor = Pipeline([("selector", sel), ("model", base)])
    else:
        predictor = Pipeline([("preprocessor", pre), ("selector", sel), ("model", base)])

names = getattr(predictor, "feature_names_in_", None)
if names is None:
    names = getattr(base, "feature_names_in_", None)
if names is None:
    block = set(targets) | excluded
    if group_col:
        block.add(group_col)
    names = [c for c in test.columns if c not in block]
X = test[list(names)]
proba = predictor.predict_proba(X)
classes = np.asarray(getattr(predictor, "classes_", getattr(base, "classes_", None)))
y = test[target].to_numpy()

pos_label = task.get("positive_label", protocol.get("positive_label"))
if pos_label is None:
    pos_label = classes[1]
matches = [c for c in classes if str(c) == str(pos_label)]
if matches:
    pos_label = matches[0]
elif pos_label not in list(classes):
    pos_label = classes[1]

if task_type == "binary":
    pos_idx = list(classes).index(pos_label)
    p = proba[:, pos_idx]
    yb = (y == pos_label).astype(int)
    pred_bin = (p >= float(threshold)).astype(int)
    neg_idx = 1 - pos_idx
    pred_label = np.where(pred_bin == 1, classes[pos_idx], classes[neg_idx])

    def rate(num, den):
        return num / den if den else np.nan

    fns = {
        "roc_auc": lambda yy, pp, pb: roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan,
        "pr_auc": lambda yy, pp, pb: average_precision_score(yy, pp),
        "balanced_accuracy": lambda yy, pp, pb: balanced_accuracy_score(yy, pb),
        "brier_score": lambda yy, pp, pb: brier_score_loss(yy, pp),
        "sensitivity": lambda yy, pp, pb: rate(np.sum((yy == 1) & (pb == 1)), np.sum(yy == 1)),
        "specificity": lambda yy, pp, pb: rate(np.sum((yy == 0) & (pb == 0)), np.sum(yy == 0)),
    }
    values = {k: f(yb, p, pred_bin) for k, f in fns.items()}

    def boot(fn):
        vals = []
        n = len(yb)
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            try:
                v = fn(yb[idx], p[idx], pred_bin[idx])
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        if not vals:
            return np.nan, np.nan
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    order = [metric_name] + [k for k in fns if k != metric_name]
    rows = []
    for k in order:
        lo, hi = boot(fns[k])
        rows.append({"metric": k, "value": values[k], "ci_low": lo, "ci_high": hi})
    metrics_df = pd.DataFrame(rows).round(6)
else:
    present = [j for j, c in enumerate(classes) if np.any(y == c)]
    aucs = [roc_auc_score((y == classes[j]).astype(int), proba[:, j]) for j in present]
    pred_label = classes[proba.argmax(axis=1)]
    value = float(np.mean(aucs))
    metrics_df = pd.DataFrame([
        {"metric": metric_name, "value": value, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "macro_f1", "value": f1_score(y, pred_label, average="macro", zero_division=0),
         "ci_low": np.nan, "ci_high": np.nan},
    ]).round(6)

mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with bootstrap intervals")
for _, r in metrics_df.iterrows():
    ci = (float(r["ci_low"]), float(r["ci_high"])) if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) else None
    mlca_runtime.record_metric(str(r["metric"]), float(r["value"]), "test", ci=ci)
if metric_name.lower() == "roc_auc":
    roc_val = float(metrics_df.loc[metrics_df["metric"] == metric_name, "value"].iloc[0])
    mlca_runtime.record_fact_candidate("test_roc_auc", roc_val, source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index, "partition": "test"})
out[f"y_true__{target}"] = y
out[f"y_pred__{target}"] = pred_label
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out.reset_index(drop=True), "test_predictions.parquet")

print(f"selected model: {sel_path.name}")
print(f"threshold: {threshold:.4g}")
print(f"test {metric_name}: {metrics_df.loc[metrics_df['metric'] == metric_name, 'value'].iloc[0]:.4g}")
print(f"predictions rows: {len(out)}")
