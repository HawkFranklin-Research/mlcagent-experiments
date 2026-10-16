import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix, brier_score_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

# Read protocol
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
positive_label = protocol.get('positive_label', None)

# Read validation metrics to find best model
val_df = pd.read_csv(inputs['09_validate_score_validation_24S6SA3P/validation_metrics.csv'])
val_manifest = json.loads(inputs['09_validate_score_validation_24S6SA3P/manifest.json'].read_text())

best_model_name = None
if 'metric' in val_df.columns and 'value' in val_df.columns:
    roc_rows = val_df[val_df['metric'] == 'roc_auc']
    if len(roc_rows) > 0:
        best_row = roc_rows.loc[roc_rows['value'].idxmax()]
        best_model_name = best_row['model'] if 'model' in best_row else best_row.name
elif 'roc_auc' in val_df.columns:
    best_row = val_df.loc[val_df['roc_auc'].idxmax()]
    best_model_name = best_row['model'] if 'model' in best_row else best_row.name

if best_model_name is None:
    raise ValueError("Cannot find best model from validation metrics")

# Map model name to file
if 'logistic' in best_model_name.lower():
    model_path = inputs['07_model_train_logistic_D8PWVNVS/model.pkl']
elif 'alternative' in best_model_name.lower():
    model_path = inputs['08_model_train_alternative_V3H8V8QN/alternative_model.pkl']
else:
    manifest_07 = json.loads(inputs['07_model_train_logistic_D8PWVNVS/manifest.json'].read_text())
    manifest_08 = json.loads(inputs['08_model_train_alternative_V3H8V8QN/manifest.json'].read_text())
    if best_model_name in str(manifest_07):
        model_path = inputs['07_model_train_logistic_D8PWVNVS/model.pkl']
    elif best_model_name in str(manifest_08):
        model_path = inputs['08_model_train_alternative_V3H8V8QN/alternative_model.pkl']
    else:
        raise ValueError(f"Cannot map model name {best_model_name} to a file")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

if positive_label is None:
    positive_label = model.classes_[1] if hasattr(model, 'classes_') else 1

# Load test data
test_df = pd.read_parquet(inputs['test'])

# Determine feature columns
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    drop_cols = set(targets) | set(excluded)
    if group_col:
        drop_cols.add(group_col)
    feature_cols = [c for c in test_df.columns if c not in drop_cols]

X_test = test_df[feature_cols].copy()
y_test = test_df[targets[0]].values

# Predict
y_proba = model.predict_proba(X_test)
if y_proba.shape[1] == 2:
    y_score = y_proba[:, list(model.classes_).index(positive_label)]
else:
    y_score = y_proba[:, 1]

# Get frozen threshold
threshold = None
if 'threshold' in val_manifest:
    threshold = val_manifest['threshold']
elif 'best_threshold' in val_manifest:
    threshold = val_manifest['best_threshold']
else:
    if 'metric' in val_df.columns:
        thresh_rows = val_df[val_df['metric'] == 'threshold']
        if len(thresh_rows) > 0:
            threshold = thresh_rows['value'].iloc[0]
    elif 'threshold' in val_df.columns:
        threshold = val_df['threshold'].iloc[0]

if threshold is None:
    val_data = pd.read_parquet(inputs['val'])
    if hasattr(model, 'feature_names_in_'):
        val_feature_cols = list(model.feature_names_in_)
    else:
        drop_cols = set(targets) | set(excluded)
        if group_col:
            drop_cols.add(group_col)
        val_feature_cols = [c for c in val_data.columns if c not in drop_cols]
    X_val = val_data[val_feature_cols].copy()
    y_val = val_data[targets[0]].values
    y_val_proba = model.predict_proba(X_val)
    if y_val_proba.shape[1] == 2:
        y_val_score = y_val_proba[:, list(model.classes_).index(positive_label)]
    else:
        y_val_score = y_val_proba[:, 1]
    y_val_binary = (y_val == positive_label).astype(int)
    thresholds = np.linspace(0, 1, 101)
    best_bal = 0
    for t in thresholds:
        y_val_pred = (y_val_score >= t).astype(int)
        bal = balanced_accuracy_score(y_val_binary, y_val_pred)
        if bal > best_bal:
            best_bal = bal
            threshold = t

# Compute metrics
y_test_binary = (y_test == positive_label).astype(int)
roc_auc = roc_auc_score(y_test_binary, y_score)
pr_auc = average_precision_score(y_test_binary, y_score)
brier = brier_score_loss(y_test_binary, y_score)

y_pred = (y_score >= threshold).astype(int)
bal_acc = balanced_accuracy_score(y_test_binary, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Bootstrap CI for ROC AUC
n_bootstraps = 1000
rng = np.random.RandomState(42)
boot_aucs = []
for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test_binary), len(y_test_binary))
    if len(np.unique(y_test_binary[indices])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_test_binary[indices], y_score[indices]))
ci_low = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)

# Save metrics
metrics_df = pd.DataFrame([
    {'metric': 'roc_auc', 'value': roc_auc, 'ci_low': ci_low, 'ci_high': ci_high},
    {'metric': 'pr_auc', 'value': pr_auc, 'ci_low': np.nan, 'ci_high': np.nan},
    {'metric': 'balanced_accuracy', 'value': bal_acc, 'ci_low': np.nan, 'ci_high': np.nan},
    {'metric': 'sensitivity', 'value': sensitivity, 'ci_low': np.nan, 'ci_high': np.nan},
    {'metric': 'specificity', 'value': specificity, 'ci_low': np.nan, 'ci_high': np.nan},
    {'metric': 'brier_score', 'value': brier, 'ci_low': np.nan, 'ci_high': np.nan},
]).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics')

# Record
mlca_runtime.record_metric('roc_auc', roc_auc, 'test', ci=(ci_low, ci_high))
mlca_runtime.record_fact_candidate('test_roc_auc', roc_auc, source='test_metrics.csv')

print(f"Test ROC AUC: {roc_auc:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})")
print(f"Test PR AUC: {pr_auc:.4f}")
print(f"Balanced accuracy: {bal_acc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
print(f"Brier score: {brier:.4f}")
