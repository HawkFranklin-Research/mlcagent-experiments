import json
import pickle
from pathlib import Path

import mlca_runtime
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())


def req(d, key, ctx):
    if key not in d:
        raise KeyError(f"Missing {key} in {ctx}; available keys: {list(d.keys())}")
    return d[key]


targets = req(prot, "targets", "_protocol")
excluded = req(prot, "excluded_columns", "_protocol")
task = req(prot, "task", "_protocol")
primary = req(task, "primary_metric", "_protocol['task']")
group = prot.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)
feature_cols = []
for c in train.columns:
    if c in exclude:
        continue
    lc = c.lower()
    if any(t in lc for t in ("id", "key", "code", "index", "idx", "row")):
        continue
    if is_string_dtype(train[c]) and train[c].is_unique:
        continue
    feature_cols.append(c)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def has_steps(o):
    return isinstance(o, Pipeline) and len(o.named_steps) > 1


def input_cols_of(o, default):
    if hasattr(o, "feature_names_in_"):
        return list(o.feature_names_in_)
    if isinstance(o, Pipeline):
        for _, step in o.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return default


selected_path = inp["latest_model"]
try:
    vm = pd.read_csv(inp["09_validate_score_validation_YWS0YQRQ/validation_metrics.csv"])
    mcol = next((c for c in vm.columns if c.lower() in ("model", "model_name", "candidate", "estimator")), None)
    metcol = next((c for c in vm.columns if c.lower() in ("metric", "metric_name")), None)
    vcol = next((c for c in vm.columns if c.lower() in ("value", "score")), None)
    if mcol and metcol and vcol:
        rows = vm[vm[metcol].astype(str).str.contains(primary, case=False, na=False)]
        scol = next((c for c in vm.columns if c.lower() in ("split", "partition")), None)
        if scol:
            rows = rows[rows[scol].astype(str).str.lower().isin(["val", "validation"])]
        if len(rows):
            best = rows.loc[rows[vcol].idxmax()]
            mn = str(best[mcol]).lower()
            if "alternative" in mn or "08" in mn:
                selected_path = inp["08_model_train_alternative_JA5Q5D9S/alternative_model.pkl"]
            elif "logistic" in mn or "07" in mn:
                selected_path = inp["07_model_train_logistic_0BAYN51T/model.pkl"]
except Exception:
    pass

pre = load_pkl(inp["05_preprocess_fit_preprocessing_M7A7QE51/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_8STHQFS1/selector.pkl"])
model = load_pkl(selected_path)

if has_steps(model):
    predictor = model
elif has_steps(pre) and any("select" in n.lower() for n in pre.named_steps):
    predictor = Pipeline([("preprocessor", pre), ("model", model)])
elif has_steps(sel):
    predictor = Pipeline([("selector", sel), ("model", model)])
else:
    predictor = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

input_cols = input_cols_of(predictor, feature_cols)
X_test = test[input_cols]
proba = predictor.predict_proba(X_test)
y_pred = predictor.predict(X_test)
final = predictor.steps[-1][1] if isinstance(predictor, Pipeline) else predictor
classes = np.asarray(final.classes_)
y_true = test[target].to_numpy()


def macro_ovr_auc(y, p, cl):
    present = [j for j, c in enumerate(cl) if np.any(y == c)]
    vals = []
    for j in present:
        yb = (y == cl[j]).astype(int)
        if len(np.unique(yb)) > 1:
            vals.append(roc_auc_score(yb, p[:, j]))
    return float(np.mean(vals)) if vals else np.nan


missing = [c for c in classes if not np.any(y_true == c)]
if missing:
    print(f"test missing classes: {missing}")

auc = macro_ovr_auc(y_true, proba, classes)
f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

rng = np.random.default_rng(42)
auc_b, f1_b = [], []
n = len(y_true)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yt, pp, yp = y_true[idx], proba[idx], y_pred[idx]
    a = macro_ovr_auc(yt, pp, classes)
    if np.isfinite(a):
        auc_b.append(a)
    f1_b.append(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))


def ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


auc_ci = ci(auc_b)
f1_ci = ci(f1_b)


def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan


metrics = pd.DataFrame([
    {"metric": primary, "value": sig4(auc), "ci_low": sig4(auc_ci[0]), "ci_high": sig4(auc_ci[1])},
    {"metric": "macro_f1", "value": sig4(f1), "ci_low": sig4(f1_ci[0]), "ci_high": sig4(f1_ci[1])},
])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, float(auc), "test", ci=auc_ci)
mlca_runtime.record_metric("macro_f1", float(f1), "test", ci=f1_ci)
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

pred = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": y_true,
    "y_pred": y_pred,
})
for j, cls in enumerate(classes):
    pred[f"proba__{cls}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print(f"test rows={len(test)} features={len(feature_cols)} model={Path(selected_path).name}")
print(f"{primary}={auc:.4f} CI=[{auc_ci[0]:.4f},{auc_ci[1]:.4f}] macro_f1={f1:.4f}")
