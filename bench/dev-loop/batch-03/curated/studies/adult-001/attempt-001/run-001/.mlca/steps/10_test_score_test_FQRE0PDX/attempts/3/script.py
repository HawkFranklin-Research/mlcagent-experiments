import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
pos_label = protocol.get('positive_label') or task.get('positive_label')

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)

test = pd.read_parquet(inputs['test'])

def get_feature_columns(model, df, protocol):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    drop = set(protocol.get('targets', [])) | set(protocol.get('excluded_columns', []))
    if protocol.get('group_column'):
        drop.add(protocol['group_column'])
    return [c for c in df.columns if c not in drop]

feat_cols = get_feature_columns(model, test, protocol)
X_test = test[feat_cols].copy()
y_test = test[targets[0]].copy()

def get_classes(model):
    if hasattr(model, 'classes_'):
        return model.classes_
    if hasattr(model, 'named_steps'):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, 'classes_'):
                return step.classes_
    raise ValueError("Cannot find classes_ in model")

classes = get_classes(model)
if pos_label is None:
    pos_label = classes[1]
pos_idx = list(classes).index(pos_label)
y_bin = (y_test == pos_label).astype(int)
y_prob = model.predict_proba(X_test)[:, pos_idx]

def get_threshold(inputs, protocol):
    val_path = inputs.get('09_validate_score_validation_N6C8NZXN/validation_metrics.csv')
    if val_path:
        df = pd.read_csv(val_path)
        if 'threshold' in df.columns:
            return float(df['threshold'].iloc[0])
        if 'metric' in df.columns and 'value' in df.columns:
            row = df[df['metric'].str.lower() == 'threshold']
            if not row.empty:
                return float(row['value'].iloc[0])
    man_path = inputs.get('09_validate_score_validation_N6C8NZXN/manifest.json')
    if man_path:
        man = json.loads(man_path.read_text())
        for key in ['threshold', 'best_threshold', 'selected_threshold', 'frozen_threshold']:
            if key in man:
                return float(man[key])
    return None

threshold = get_threshold(inputs, protocol)
if threshold is None:
    threshold = getattr(model, 'threshold', 0.5)

roc_auc = roc_auc_score(y_bin, y_prob)
pr_auc = average_precision_score(y_bin, y_prob)
y_pred = (y_prob >= threshold).astype(int)
bal_acc = balanced_accuracy_score(y_bin, y_pred)
brier = brier_score_loss(y_bin, y_prob)
tn, fp, fn, tp = confusion_matrix(y_bin, y_pred, labels=[0, 1]).ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

rng = np.random.default_rng(seed)
n = len(y_bin)
boot = {k: [] for k in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier', 'sensitivity', 'specificity']}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yt = y_bin.values[idx]
    yp = y_prob[idx]
    try:
        boot['roc_auc'].append(roc_auc_score(yt, yp))
    except ValueError:
        boot['roc_auc'].append(np.nan)
    try:
        boot['pr_auc'].append(average_precision_score(yt, yp))
    except ValueError:
        boot['pr_auc'].append(np.nan)
    yp_pred = (yp >= threshold).astype(int)
    try:
        boot['balanced_accuracy'].append(balanced_accuracy_score(yt, yp_pred))
    except ValueError:
        boot['balanced_accuracy'].append(np.nan)
    try:
        boot['brier'].append(brier_score_loss(yt, yp))
    except ValueError:
        boot['brier'].append(np.nan)
    tn, fp, fn, tp = confusion_matrix(yt, yp_pred, labels=[0, 1]).ravel()
    boot['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    boot['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

rows = []
for name, val in [('roc_auc', roc_auc), ('pr_auc', pr_auc), ('balanced_accuracy', bal_acc), ('brier', brier), ('sensitivity', sens), ('specificity', spec)]:
    ci_low = np.nanpercentile(boot[name], 2.5)
    ci_high = np.nanpercentile(boot[name], 97.5)
    rows.append({'metric': name, 'value': val, 'ci_low': ci_low, 'ci_high': ci_high})
df_metrics = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df_metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

for _, row in df_metrics.iterrows():
    mlca_runtime.record_metric(row['metric'], row['value'], split='test', ci=(row['ci_low'], row['ci_high']))
mlca_runtime.record_fact_candidate('test_roc_auc', roc_auc, source='test_metrics.csv')

print(f"Test rows: {len(test)}")
print(f"Primary metric {primary_metric}: {roc_auc:.4f}")
