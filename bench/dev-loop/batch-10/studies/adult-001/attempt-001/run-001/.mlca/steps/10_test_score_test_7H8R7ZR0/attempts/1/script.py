import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, recall_score, f1_score, roc_curve

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column']:
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available top-level keys: {list(protocol.keys())}")
task = protocol['task']
targets = protocol['targets']
target_col = targets[0] if isinstance(targets, list) else targets
primary_metric = task['primary_metric']
task_type = task.get('type', 'binary')
positive_label = task.get('positive_label', protocol.get('positive_label', None))

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_features(model):
    feats = getattr(model, 'feature_names_in_', None)
    if feats is None and hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            feats = getattr(step, 'feature_names_in_', None)
            if feats is not None:
                break
    if feats is None:
        raise ValueError("Model lacks feature_names_in_")
    return list(feats)

def predict_proba(model, df):
    return model.predict_proba(df[get_features(model)])

def macro_ovr_auc(y_true, proba, classes):
    aucs = []
    for j, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aucs.append(roc_auc_score(y_bin, proba[:, j]))
    return np.mean(aucs) if aucs else np.nan

def compute_metrics(y_true, proba, classes, threshold, task_type):
    m = {}
    if task_type == 'binary':
        pos = positive_label if positive_label is not None else classes[1]
        pos_idx = list(classes).index(pos)
        p = proba[:, pos_idx]
        y_bin = (y_true == pos).astype(int)
        m['roc_auc'] = roc_auc_score(y_bin, p)
        m['pr_auc'] = average_precision_score(y_bin, p)
        m['brier_score'] = brier_score_loss(y_bin, p)
        y_pred = (p >= threshold).astype(int)
        m['balanced_accuracy'] = balanced_accuracy_score(y_bin, y_pred)
        m['sensitivity'] = recall_score(y_bin, y_pred, pos_label=1, zero_division=0)
        m['specificity'] = recall_score(y_bin, y_pred, pos_label=0, zero_division=0)
    else:
        m['roc_auc_ovr'] = macro_ovr_auc(y_true, proba, classes)
        y_pred = classes[np.argmax(proba, axis=1)]
        m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return m

val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])
model_paths = {
    'logistic': inputs['07_model_train_logistic_23SCN2HD/model.pkl'],
    'alternative': inputs['08_model_train_alternative_9VT6FVM1/alternative_model.pkl'],
}
models = {name: load_pkl(path) for name, path in model_paths.items()}
scores = {}
for name, model in models.items():
    proba = predict_proba(model, val)
    scores[name] = compute_metrics(val[target_col].values, proba, model.classes_, 0.5, task_type)[primary_metric]
direction = 1 if primary_metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc_ovr', 'f1_macro'] else -1
best_name = max(scores, key=scores.get) if direction == 1 else min(scores, key=scores.get)
model = models[best_name]
classes = model.classes_

threshold = None
if task_type == 'binary':
    proba_val = predict_proba(model, val)
    pos = positive_label if positive_label is not None else classes[1]
    p_val = proba_val[:, list(classes).index(pos)]
    y_val_bin = (val[target_col].values == pos).astype(int)
    fpr, tpr, thr = roc_curve(y_val_bin, p_val)
    threshold = thr[np.argmax(tpr - fpr)]
    if not np.isfinite(threshold):
        threshold = 0.5

proba_test = predict_proba(model, test)
y_test = test[target_col].values
point = compute_metrics(y_test, proba_test, classes, threshold, task_type)
rng = np.random.default_rng(42)
n = len(y_test)
boot = {k: [] for k in point}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    try:
        m = compute_metrics(y_test[idx], proba_test[idx], classes, threshold, task_type)
    except Exception:
        continue
    for k, v in m.items():
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for k, v in point.items():
    arr = np.array(boot[k])
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
    rows.append({'metric': k, 'value': v, 'ci_low': ci_low, 'ci_high': ci_high})
metrics_df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', 'Test metrics with 95% bootstrap CI')

row = metrics_df[metrics_df['metric'] == primary_metric].iloc[0]
mlca_runtime.record_metric(primary_metric, float(row['value']), 'test', ci=(float(row['ci_low']), float(row['ci_high'])))
if primary_metric == 'roc_auc':
    roc_row = metrics_df[metrics_df['metric'] == 'roc_auc'].iloc[0]
    mlca_runtime.record_fact_candidate('test_roc_auc', float(roc_row['value']), source='test_metrics.csv')
print(f"Selected {best_name}, threshold={threshold if threshold is None else round(threshold,4)}")
print(f"Test {primary_metric}={point[primary_metric]:.4f}, n_test={n}")
