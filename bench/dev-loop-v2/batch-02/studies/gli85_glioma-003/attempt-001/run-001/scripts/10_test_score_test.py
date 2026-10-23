import mlca_runtime, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    metric = task["primary_metric"]
    targets = protocol["targets"]
    excluded = protocol.get("excluded_columns", [])
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(protocol.keys())}")
target = targets[0]
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

def keylike(s):
    if s.name in targets or s.name in excluded or (group_col is not None and s.name == group_col):
        return False
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not keylike(train[c])]

model_paths = {
    "logistic": inp["07_model_train_logistic_9GB0B8SE/model.pkl"],
    "alternative": inp["08_model_train_alternative_C89NP2WV/alternative_model.pkl"],
}
selected_path = inp["latest_model"]
try:
    vm = pd.read_csv(inp["09_validate_score_validation_Y9PBCTB4/validation_metrics.csv"])
    name_col = next((c for c in vm.columns if c.lower() in ("model", "model_name", "candidate", "estimator")), None)
    metric_col = next((c for c in vm.columns if c.lower() == metric.lower() or c.lower().replace(" ", "_") == metric.lower()), None)
    if name_col and metric_col:
        higher = metric.lower() not in ("brier_score", "brier", "log_loss", "mae", "rmse")
        idx = vm[metric_col].idxmax() if higher else vm[metric_col].idxmin()
        name = str(vm.loc[idx, name_col]).lower()
        for k, p in model_paths.items():
            if k in name or name in k:
                selected_path = p
                break
except Exception:
    pass

def load(p):
    return pickle.loads(Path(p).read_bytes())

pre = load(inp["05_preprocess_fit_preprocessing_W6YRPZJC/preprocessor.pkl"])
sel = load(inp["06_features_select_features_GPEMXWYC/selector.pkl"])
model = load(selected_path)

def frontend(pre, sel):
    if isinstance(sel, Pipeline) and any(k in sel.named_steps for k in ("preprocessor", "preprocess", "pre")):
        return sel
    if isinstance(pre, Pipeline) and any(k in pre.named_steps for k in ("selector", "feature_selection")):
        return pre
    return Pipeline([("preprocessor", pre), ("selector", sel)])

if isinstance(model, Pipeline) and len(model.named_steps) > 1:
    pipe = model
else:
    front = frontend(pre, sel)
    if isinstance(model, Pipeline):
        est = list(model.named_steps.values())[0]
        pipe = Pipeline([("frontend", front), ("model", est)])
    else:
        pipe = Pipeline([("frontend", front), ("model", model)])

if hasattr(pipe, "feature_names_in_"):
    input_cols = list(pipe.feature_names_in_)
elif hasattr(pre, "feature_names_in_"):
    input_cols = list(pre.feature_names_in_)
elif hasattr(sel, "feature_names_in_"):
    input_cols = list(sel.feature_names_in_)
else:
    input_cols = feature_cols

classes = list(pipe.classes_)
pos_label = task.get("positive_label")
if pos_label is None or pos_label not in classes:
    pos_label = classes[1]
pos_idx = classes.index(pos_label)

val_proba = pipe.predict_proba(val[input_cols])[:, pos_idx]
val_y_bin = (val[target] == pos_label).astype(int).to_numpy()
thr_grid = np.unique(np.concatenate([val_proba, [0.5]]))
best_thr, best_ba = 0.5, -1.0
for t in thr_grid:
    ba = balanced_accuracy_score(val_y_bin, (val_proba >= t).astype(int))
    if ba > best_ba:
        best_ba, best_thr = ba, t
threshold = float(best_thr)

proba = pipe.predict_proba(test[input_cols])
proba_pos = proba[:, pos_idx]
y_true = test[target].to_numpy()
y_bin = (test[target] == pos_label).astype(int).to_numpy()
y_pred_bin = (proba_pos >= threshold).astype(int)
neg_label = classes[0] if pos_label == classes[1] else classes[1]
y_pred = np.where(y_pred_bin == 1, pos_label, neg_label)

def point_metrics(yb, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else np.nan,
        "pr_auc": average_precision_score(yb, p),
        "balanced_accuracy": balanced_accuracy_score(yb, yp),
        "brier_score": brier_score_loss(yb, p),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    }

metrics_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
point = point_metrics(y_bin, proba_pos, threshold)
boot = {m: [] for m in metrics_names}
rng = np.random.default_rng(42)
n = len(y_bin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb = y_bin[idx], proba_pos[idx]
    for m in metrics_names:
        if m == "roc_auc":
            if len(np.unique(yb)) < 2:
                continue
            v = roc_auc_score(yb, pb)
        elif m == "pr_auc":
            v = average_precision_score(yb, pb)
        elif m == "balanced_accuracy":
            v = balanced_accuracy_score(yb, (pb >= threshold).astype(int))
        elif m == "brier_score":
            v = brier_score_loss(yb, pb)
        else:
            yp = (pb >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
            v = tp / (tp + fn) if m == "sensitivity" and (tp + fn) > 0 else (tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        if not np.isnan(v):
            boot[m].append(v)

def sig4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + 3)

rows = []
for m in metrics_names:
    lo, hi = (np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)) if boot[m] else (np.nan, np.nan)
    rows.append({"metric": m, "value": sig4(point[m]), "ci_low": sig4(lo), "ci_high": sig4(hi)})
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

primary = point[metric]
primary_lo, primary_hi = (np.percentile(boot[metric], 2.5), np.percentile(boot[metric], 97.5)) if boot[metric] else (np.nan, np.nan)
mlca_runtime.record_metric(metric, primary, "test", ci=(primary_lo, primary_hi))
if "roc_auc" in point and not np.isnan(point["roc_auc"]):
    mlca_runtime.record_fact_candidate("test_roc_auc", point["roc_auc"], source="test_metrics.csv")

out = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y_true, "y_pred": y_pred})
for j, c in enumerate(classes):
    out[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

print(f"test rows: {len(test)}")
print(f"{metric}: {primary:.4f} CI [{primary_lo:.4f}, {primary_hi:.4f}]")
