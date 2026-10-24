import mlca_runtime
import pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
primary_metric = protocol["task"]["primary_metric"]
target = protocol["targets"][0]
positive_label = protocol["task"].get("positive_label", None)

test_df = pd.read_parquet(paths["test"])

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    with open(paths["06_preprocess_fit_preprocessing_Z6Z720YG/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(paths["07_features_select_features_Y69JECM8/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    model = Pipeline([("pre", pre), ("sel", sel), ("clf", model)])
    feature_cols = list(model.feature_names_in_)

X_test = test_df[feature_cols]
y_true = test_df[target].values
proba = model.predict_proba(X_test)
classes = list(model.classes_)
pos_idx = classes.index(positive_label) if positive_label is not None else 1
p_pos = proba[:, pos_idx]
y_bin = (y_true == classes[pos_idx]).astype(int)

threshold = 0.5
vman = json.loads(paths["10_validate_score_validation_BN5FNS8Y/manifest.json"].read_text())
for k in ("threshold", "selected_threshold", "best_threshold", "youden_threshold"):
    if k in vman:
        threshold = float(vman[k])
        break

def boot_ci(fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, size=n_rows)
        try:
            v = float(fn(idx))
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    arr = np.asarray(vals)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def f_roc(idx): return roc_auc_score(y_bin[idx], p_pos[idx])
def f_pr(idx): return average_precision_score(y_bin[idx], p_pos[idx])
def f_bal(idx):
    pred = (p_pos[idx] >= threshold).astype(int)
    return balanced_accuracy_score(y_bin[idx], pred)
def f_brier(idx): return brier_score_loss(y_bin[idx], p_pos[idx])
def f_sens(idx):
    pred = (p_pos[idx] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin[idx], pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan
def f_spec(idx):
    pred = (p_pos[idx] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin[idx], pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

all_idx = np.arange(len(y_true))
metrics = {}
for name, fn in [("roc_auc", f_roc), ("pr_auc", f_pr), ("balanced_accuracy", f_bal),
                 ("brier_score", f_brier), ("sensitivity", f_sens), ("specificity", f_spec)]:
    v = float(fn(all_idx))
    lo, hi = boot_ci(fn)
    metrics[name] = (v, lo, hi)

y_pred = np.where(p_pos >= threshold, classes[pos_idx], classes[1 - pos_idx])

rows = [{"metric": k, "value": v[0], "ci_low": v[1], "ci_high": v[2]}
        for k, v in metrics.items()]
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv",
                        caption="Test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary_metric, metrics[primary_metric][0], "test",
                           ci=(metrics[primary_metric][1], metrics[primary_metric][2]))
mlca_runtime.record_fact_candidate("test_roc_auc", metrics["roc_auc"][0],
                                   source="test_metrics.csv")

out_df = pd.DataFrame({
    "row_id": test_df.index,
    "partition": "test",
    "y_true": y_true,
    "y_pred": y_pred,
})
for i, c in enumerate(classes):
    out_df[f"proba__{c}"] = proba[:, i]
mlca_runtime.save_row_outputs(out_df, "test_predictions.parquet")

print(f"n_test={len(test_df)} primary={primary_metric}")
print(f"roc_auc={metrics['roc_auc'][0]:.4f} thr={threshold:.4f}")
