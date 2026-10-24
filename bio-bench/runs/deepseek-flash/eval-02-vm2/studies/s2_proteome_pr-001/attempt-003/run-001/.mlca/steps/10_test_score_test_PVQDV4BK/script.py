import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score)

inputs = mlca_runtime.inputs()
mlca_runtime.set_seed(42)
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
task_type = str(task.get("type", "")).lower()

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
test = pd.read_parquet(inputs["test"])

feature_cols = []
for c in train.columns:
    if c in set(targets) or c in excluded or (group and c == group):
        continue
    s = train[c]
    if s.nunique(dropna=False) == len(s) and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        continue
    feature_cols.append(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_GK4CB9FZ/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_5542Z2W2/selector.pkl"])
model_log = load_pkl(inputs["07_model_train_logistic_VYQ2HW60/model.pkl"])
model_alt = load_pkl(inputs["08_model_train_alternative_J88G12HR/alternative_model.pkl"])
latest = load_pkl(inputs["latest_model"])

with open(inputs["09_validate_score_validation_XS6CDKH6/manifest.json"]) as f:
    vman = json.load(f)

def find_key(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names:
                return v
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None

sel_name = find_key(vman, {"selected_model", "chosen_model", "best_model", "model_path", "model_name", "model", "estimator"})
if isinstance(sel_name, dict):
    sel_name = sel_name.get("path") or sel_name.get("name") or str(sel_name)

if sel_name is None:
    vm = pd.read_csv(inputs["09_validate_score_validation_XS6CDKH6/validation_metrics.csv"])
    cl = {str(c).lower(): c for c in vm.columns}
    mcol = next((cl[k] for k in ["model", "candidate", "model_name", "estimator", "name"] if k in cl), None)
    metcol = next((cl[k] for k in ["metric", "metric_name"] if k in cl), None)
    valcol = next((cl[k] for k in ["value", "score", "metric_value"] if k in cl), None)
    lower_better = any(x in primary.lower() for x in ["brier", "loss", "error", "mse", "mae"])
    if mcol and metcol and valcol:
        sub = vm[vm[metcol].astype(str).str.lower() == primary.lower()]
        if len(sub):
            sel_name = str(sub.sort_values(valcol, ascending=lower_better).iloc[0][mcol])
    elif mcol and primary in vm.columns:
        sel_name = str(vm.sort_values(primary, ascending=lower_better).iloc[0][mcol])

sname = str(sel_name).lower() if sel_name is not None else ""
if "alt" in sname or "08" in sname or "alternative" in sname:
    model_obj = model_alt
elif "log" in sname or "07" in sname:
    model_obj = model_log
else:
    model_obj = latest

def build_model(m):
    m_in = list(getattr(m, "feature_names_in_", []))
    m_steps = getattr(m, "named_steps", {})
    if m_in and set(m_in).issubset(set(feature_cols)):
        return m
    if len(m_steps) > 1:
        return Pipeline([("preprocessor", pre), ("model", m)])
    pre_has_sel = hasattr(pre, "named_steps") and any("select" in n.lower() for n in pre.named_steps)
    sel_has_pre = hasattr(sel, "named_steps") and any("preprocess" in n.lower() for n in sel.named_steps)
    steps = []
    if not sel_has_pre:
        steps.append(("preprocessor", pre))
    if not pre_has_sel:
        steps.append(("selector", sel))
    steps.append(("model", m))
    return Pipeline(steps)

final_model = build_model(model_obj)
classes = np.asarray(final_model.classes_)
classes_list = list(classes)

def X_for(m, df):
    cols = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else feature_cols
    return df[cols]

thr = find_key(vman, {"threshold", "decision_threshold", "best_threshold", "cutoff", "optimal_threshold"})

y_test = test[target].values
X_test = X_for(final_model, test)
proba = final_model.predict_proba(X_test)

if task_type == "binary":
    pos_label = task.get("positive_label") or protocol.get("positive_label")
    if pos_label is None:
        pos_label = classes[1]
    pos_idx = int(np.where(classes == pos_label)[0][0])
    neg_label = classes[1 - pos_idx]
    p = proba[:, pos_idx]
    if thr is None:
        pv = final_model.predict_proba(X_for(final_model, val))[:, pos_idx]
        yv = val[target].values
        best_thr, best_ba = 0.5, -1.0
        for t in np.unique(np.r_[0.0, pv, 1.0]):
            yp = np.where(pv >= t, pos_label, neg_label)
            ba = balanced_accuracy_score(yv, yp)
            if ba > best_ba:
                best_ba, best_thr = ba, float(t)
        thr = best_thr
    thr = float(thr)
    def binary_metrics(y, pv, t):
        yb = (y == pos_label).astype(int)
        ypb = (pv >= t).astype(int)
        yp = np.where(ypb == 1, pos_label, neg_label)
        out = {}
        try: out["roc_auc"] = roc_auc_score(yb, pv)
        except Exception: out["roc_auc"] = np.nan
        try: out["pr_auc"] = average_precision_score(yb, pv)
        except Exception: out["pr_auc"] = np.nan
        out["balanced_accuracy"] = balanced_accuracy_score(y, yp)
        try: out["brier_score"] = brier_score_loss(yb, pv)
        except Exception: out["brier_score"] = np.nan
        tn, fp, fn, tp = confusion_matrix(yb, ypb, labels=[0, 1]).ravel()
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
        return out
    metrics = binary_metrics(y_test, p, thr)
    y_pred = np.where(p >= thr, pos_label, neg_label)
    p_arr = p
    metric_fun = lambda yb, pb: binary_metrics(yb, pb, thr)
else:
    if primary not in ["roc_auc_ovr", "roc_auc", "f1_macro", "f1"]:
        raise KeyError(f"unsupported multiclass primary_metric {primary}")
    def multi_metrics(y, pr):
        out = {}
        try:
            out["roc_auc_ovr"] = roc_auc_score(y, pr, multi_class="ovr", average="macro", labels=classes)
        except Exception:
            vals = []
            present = [c for c in classes_list if c in set(y)]
            for c in present:
                j = classes_list.index(c)
                try: vals.append(roc_auc_score((y == c).astype(int), pr[:, j]))
                except Exception: pass
            out["roc_auc_ovr"] = float(np.mean(vals)) if vals else np.nan
        yp = classes[np.argmax(pr, axis=1)]
        out["f1_macro"] = f1_score(y, yp, average="macro")
        out["roc_auc"] = out["roc_auc_ovr"]
        out["f1"] = out["f1_macro"]
        return out
    metrics = multi_metrics(y_test, proba)
    y_pred = classes[np.argmax(proba, axis=1)]
    p_arr = proba
    metric_fun = multi_metrics

if primary not in metrics:
    raise KeyError(f"primary metric {primary} not computed; available: {list(metrics)}")

rng = np.random.default_rng(42)
n = len(test)
boot = {k: [] for k in metrics}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb = y_test[idx]
    pb = p_arr[idx]
    mb = metric_fun(yb, pb)
    for k in metrics:
        boot[k].append(mb.get(k, np.nan))

ci = {}
for k in metrics:
    vals = np.asarray(boot[k], dtype=float)
    ci[k] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))

def r4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

rows = []
for k in [primary] + [k for k in metrics if k != primary]:
    lo, hi = ci[k]
    rows.append({"metric": k, "value": r4(metrics[k]), "ci_low": r4(lo), "ci_high": r4(hi)})
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary, float(metrics[primary]), "test", ci=(ci[primary][0], ci[primary][1]))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(metrics["roc_auc"]), source="test_metrics.csv")

pred = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y_test, "y_pred": y_pred})
for j, c in enumerate(classes_list):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print("test rows", len(test), "feature cols", len(feature_cols))
print("selected", str(sel_name)[:80], "threshold", thr)
print(primary, r4(metrics[primary]), "ci", (r4(ci[primary][0]), r4(ci[primary][1])))
