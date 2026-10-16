import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
n_classes = task.get("n_classes", None)

train = pd.read_parquet(inputs["train"])
all_cols = list(train.columns)
drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)

def is_key_like(series, name):
    if series.dtype == 'object' or series.dtype == 'string':
        return series.nunique() == len(series)
    if pd.api.types.is_integer_dtype(series):
        if series.nunique() == len(series):
            if any(kw in name.lower() for kw in ['id', 'code', 'key', 'index']):
                return True
    return False

feature_cols = [c for c in all_cols if c not in drop_cols and not is_key_like(train[c], c)]

manifest_09 = json.loads(inputs["09_validate_score_validation_ZZY7R6SH/manifest.json"].read_text())
model_file = None
for key in ["model", "model_path", "chosen_model", "frozen_model", "best_model"]:
    if key in manifest_09:
        val = manifest_09[key]
        if isinstance(val, str):
            if val in inputs:
                model_file = inputs[val]
                break
            for k, v in inputs.items():
                if v.name == val or k.endswith(val):
                    model_file = v
                    break
        elif isinstance(val, dict):
            for k2 in ["path", "file", "model_path"]:
                if k2 in val:
                    model_file = inputs.get(val[k2], None)
                    if model_file: break
        if model_file: break

if model_file is None:
    for key in ["frozen_model", "chosen_model", "model"]:
        if key in protocol:
            val = protocol[key]
            if isinstance(val, str):
                if val in inputs:
                    model_file = inputs[val]
                    break
                for k, v in inputs.items():
                    if v.name == val or k.endswith(val):
                        model_file = v
                        break

if model_file is None:
    def get_val_auc(manifest):
        for k in ["validation_metrics", "metrics", "val_metrics"]:
            if k in manifest:
                m = manifest[k]
                if isinstance(m, dict):
                    for auc_key in ["roc_auc_ovr", "roc_auc", "val_roc_auc"]:
                        if auc_key in m:
                            return m[auc_key]
                elif isinstance(m, list):
                    for row in m:
                        if isinstance(row, dict) and row.get("metric") in ["roc_auc_ovr", "roc_auc"]:
                            return row.get("value")
        for auc_key in ["roc_auc_ovr", "roc_auc", "val_roc_auc"]:
            if auc_key in manifest:
                return manifest[auc_key]
        return None
    manifest_07 = json.loads(inputs["07_model_train_logistic_8D243SEH/manifest.json"].read_text())
    manifest_08 = json.loads(inputs["08_model_train_alternative_052PRAHM/manifest.json"].read_text())
    auc_07 = get_val_auc(manifest_07)
    auc_08 = get_val_auc(manifest_08)
    if auc_07 is not None and auc_08 is not None:
        model_file = inputs["07_model_train_logistic_8D243SEH/model.pkl"] if auc_07 >= auc_08 else inputs["08_model_train_alternative_052PRAHM/alternative_model.pkl"]
    elif auc_07 is not None:
        model_file = inputs["07_model_train_logistic_8D243SEH/model.pkl"]
    elif auc_08 is not None:
        model_file = inputs["08_model_train_alternative_052PRAHM/alternative_model.pkl"]
    else:
        model_file = inputs["latest_model"]

with open(model_file, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)

test = pd.read_parquet(inputs["test"])
X_test = test[feature_cols].copy()
y_test = test[targets[0]]

y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

if n_classes and n_classes > 2:
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=model.classes_)
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    pr_auc = average_precision_score(y_test_bin, y_proba, average='macro')
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    brier = np.mean(np.sum((y_test_bin - y_proba)**2, axis=1))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sensitivities, specificities = [], []
    for i in range(len(model.classes_)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sensitivities.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        specificities.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    macro_sens = np.mean(sensitivities)
    macro_spec = np.mean(specificities)
else:
    pos_label = protocol.get("positive_label", model.classes_[1])
    pos_idx = list(model.classes_).index(pos_label)
    y_proba_pos = y_proba[:, pos_idx]
    roc_auc = roc_auc_score(y_test, y_proba_pos)
    pr_auc = average_precision_score(y_test, y_proba_pos, pos_label=pos_label)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    brier = np.mean((y_proba_pos - (y_test == pos_label).astype(int))**2)
    threshold = 0.5
    for key in ["threshold", "frozen_threshold", "best_threshold"]:
        if key in manifest_09:
            threshold = manifest_09[key]
            break
    val_metrics = pd.read_csv(inputs["09_validate_score_validation_ZZY7R6SH/validation_metrics.csv"])
    if "threshold" in val_metrics.columns:
        threshold = val_metrics["threshold"].iloc[0]
    y_pred_thresh = (y_proba_pos >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresh, labels=[0, 1])
    macro_sens = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    macro_spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0

def bootstrap_roc_auc(y_true, y_proba, n_resamples=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        y_true_boot = y_true.iloc[idx]
        y_proba_boot = y_proba[idx]
        if n_classes and n_classes > 2:
            if len(np.unique(y_true_boot)) < n_classes:
                continue
            try:
                auc = roc_auc_score(y_true_boot, y_proba_boot, multi_class='ovr', average='macro', labels=model.classes_)
            except ValueError:
                continue
        else:
            if len(np.unique(y_true_boot)) < 2:
                continue
            auc = roc_auc_score(y_true_boot, y_proba_boot[:, pos_idx])
        aucs.append(auc)
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

ci_low, ci_high = bootstrap_roc_auc(y_test, y_proba, n_resamples=1000, seed=42)

metrics_df = pd.DataFrame([
    {"metric": "roc_auc", "value": roc_auc, "ci_low": ci_low, "ci_high": ci_high},
    {"metric": "pr_auc", "value": pr_auc, "ci_low": np.nan, "ci_high": np.nan},
    {"metric": "balanced_accuracy", "value": bal_acc, "ci_low": np.nan, "ci_high": np.nan},
    {"metric": "sensitivity", "value": macro_sens, "ci_low": np.nan, "ci_high": np.nan},
    {"metric": "specificity", "value": macro_spec, "ci_low": np.nan, "ci_high": np.nan},
    {"metric": "brier_score", "value": brier, "ci_low": np.nan, "ci_high": np.nan},
])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics")

mlca_runtime.record_metric("roc_auc", roc_auc, split="test", ci=(ci_low, ci_high))
mlca_runtime.record_fact_candidate("test_roc_auc", roc_auc, source="test_metrics.csv")

print(f"Test ROC AUC: {roc_auc:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})")
print(f"Test PR AUC: {pr_auc:.4f}")
print(f"Test balanced accuracy: {bal_acc:.4f}")
print(f"Test sensitivity: {macro_sens:.4f}, specificity: {macro_spec:.4f}")
print(f"Test Brier score: {brier:.4f}")
