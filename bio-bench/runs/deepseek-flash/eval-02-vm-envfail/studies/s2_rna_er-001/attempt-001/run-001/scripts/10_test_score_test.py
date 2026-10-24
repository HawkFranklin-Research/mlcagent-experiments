import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, confusion_matrix)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"{k} missing; available top-level keys: {sorted(d.keys())}")
    return d[k]

task = need(protocol, "task")
if "primary_metric" not in task:
    raise KeyError(f"primary_metric missing in task; top-level keys: {sorted(protocol.keys())}")
primary = task["primary_metric"]
targets = need(protocol, "targets")
excluded = set(need(protocol, "excluded_columns"))
if not isinstance(excluded, (list, tuple, set)):
    raise TypeError("excluded_columns must be a list-like column collection")
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError("This script expects exactly one declared target")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

raw_cols = [c for c in train.columns if c not in set(targets) and c not in excluded and c != group]
def keylike(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    return False
feature_cols = [c for c in raw_cols if not keylike(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

vm = pd.read_csv(inp["09_validate_score_validation_NR2XYZ44/validation_metrics.csv"])
def model_path_from_validation(vm):
    name = ""
    higher = primary not in {"brier_score", "log_loss", "mse", "mae", "rmse"}
    if "model" in vm.columns and primary in vm.columns:
        r = vm.dropna(subset=[primary]).sort_values(primary, ascending=not higher).iloc[0]
        name = str(r["model"]).lower()
    elif {"model", "metric", "value"}.issubset(vm.columns):
        sub = vm[vm["metric"] == primary]
        if len(sub):
            r = sub.sort_values("value", ascending=not higher).iloc[0]
            name = str(r["model"]).lower()
    if "alternative" in name or "alt" in name:
        return inp["08_model_train_alternative_55567D2N/alternative_model.pkl"]
    if "logistic" in name or "logit" in name:
        return inp["07_model_train_logistic_8JHDFT9S/model.pkl"]
    return inp["latest_model"]
model_path = model_path_from_validation(vm)

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)
pre = load_pkl(inp["05_preprocess_fit_preprocessing_R5N312M1/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_JQQFHDQ8/selector.pkl"])
mdl = load_pkl(model_path)

def accepts_raw(est):
    fn = getattr(est, "feature_names_in_", None)
    return fn is not None and list(fn) == feature_cols

if isinstance(mdl, Pipeline) and hasattr(mdl, "predict_proba"):
    predictor = mdl
elif accepts_raw(mdl):
    predictor = mdl
elif accepts_raw(sel):
    predictor = Pipeline([("selector", sel), ("model", mdl)])
else:
    predictor = Pipeline([("pre", pre), ("selector", sel), ("model", mdl)])

def classes_of(est):
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if isinstance(est, Pipeline):
        return classes_of(est.steps[-1][1])
    raise AttributeError("classes_ not found")
classes = classes_of(predictor)
if len(classes) != 2:
    raise ValueError("Binary evaluation requires exactly two model classes")

def proba_on(df, est):
    cols = list(est.feature_names_in_) if hasattr(est, "feature_names_in_") else feature_cols
    for c in cols:
        if c in set(targets) or c in excluded:
            raise ValueError(f"Target or excluded column reached model: {c}")
    return est.predict_proba(df[cols])

positive_label = task.get("positive_label", task.get("positive_class", classes[1]))
pos_idx = int(np.where(classes == positive_label)[0][0])
neg_label = classes[1 - pos_idx]

proba_val = proba_on(val, predictor)[:, pos_idx]
y_val_bin = (val[target].to_numpy() == positive_label).astype(int)
if y_val_bin.min() == 0 and y_val_bin.max() == 1:
    fpr, tpr, thr_grid = roc_curve(y_val_bin, proba_val)
    threshold = float(thr_grid[np.argmax(tpr - fpr)])
else:
    threshold = 0.5
if not np.isfinite(threshold):
    threshold = 0.5

proba_test = proba_on(test, predictor)
proba_pos = proba_test[:, pos_idx]
y_test = test[target].to_numpy()
y_test_bin = (y_test == positive_label).astype(int)
y_pred_bin = (proba_pos >= threshold).astype(int)
y_pred = np.where(y_pred_bin == 1, positive_label, neg_label)

def safe(f):
    try:
        return float(f())
    except Exception:
        return np.nan

def compute_metrics(yb, p, yp):
    tn, fp, fn, tp = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "roc_auc": safe(lambda: roc_auc_score(yb, p)) if len(np.unique(yb)) > 1 else np.nan,
        "pr_auc": safe(lambda: average_precision_score(yb, p)) if len(np.unique(yb)) > 1 else np.nan,
        "balanced_accuracy": safe(lambda: balanced_accuracy_score(yb, yp)),
        "brier_score": safe(lambda: brier_score_loss(yb, p)),
        "sensitivity": safe(lambda: sens),
        "specificity": safe(lambda: spec),
    }

point = compute_metrics(y_test_bin, proba_pos, y_pred_bin)
if primary not in point:
    raise ValueError(f"Declared primary_metric {primary} is not among computed binary metrics")

metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
rng = np.random.default_rng(42)
boot = {k: [] for k in metric_names}
n = len(y_test_bin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    vals = compute_metrics(y_test_bin[idx], proba_pos[idx], y_pred_bin[idx])
    for k in metric_names:
        boot[k].append(vals[k])

ci = {k: (float(np.nanpercentile(boot[k], 2.5)), float(np.nanpercentile(boot[k], 97.5))) for k in metric_names}

def r4(x):
    if x is None or not np.isfinite(float(x)):
        return np.nan
    return float(f"{float(x):.4g}")

rows = []
for k in metric_names:
    rows.append({
        "metric": primary if k == primary else k,
        "value": r4(point[k]),
        "ci_low": r4(ci[k][0]),
        "ci_high": r4(ci[k][1]),
    })
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

for row in rows:
    if np.isfinite(row["value"]):
        mlca_runtime.record_metric(row["metric"], row["value"], split="test", ci=(row["ci_low"], row["ci_high"]))
if primary == "roc_auc" and np.isfinite(point["roc_auc"]):
    mlca_runtime.record_fact_candidate("test_roc_auc", r4(point["roc_auc"]), source="test_metrics.csv")

out = pd.DataFrame({
    "row_id": test.index.to_numpy(),
    "partition": "test",
    f"y_true__{target}": y_test,
    f"y_pred__{target}": y_pred,
})
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba_test[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print(f"test rows={len(test)} features={len(feature_cols)} model={model_path.name}")
print(f"primary {primary}={r4(point[primary])} ci=({r4(ci[primary][0])},{r4(ci[primary][1])})")
print(f"threshold={r4(threshold)} positive={positive_label}")
