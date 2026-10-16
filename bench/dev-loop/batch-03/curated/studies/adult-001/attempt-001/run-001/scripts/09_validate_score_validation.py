import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task['type']
positive_label = task.get('positive_label')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
for c in list(feature_cols):
    if train[c].dtype == 'object' and train[c].nunique() == len(train):
        feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train):
        feature_cols.remove(c)

def get_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

def get_classes(model):
    if hasattr(model, 'classes_'):
        return model.classes_
    if isinstance(model, Pipeline):
        return model.steps[-1][1].classes_
    return None

models = {
    'logistic': pickle.load(open(inputs['07_model_train_logistic_CV1F3546/model.pkl'], 'rb')),
    'alternative': pickle.load(open(inputs['08_model_train_alternative_Z4FMHBXH/alternative_model.pkl'], 'rb'))
}

y_val = val[targets[0]]
rows = []
for name, model in models.items():
    cols = get_input_cols(model) or feature_cols
    X_val = val[cols]
    classes = get_classes(model)
    if task_type == 'binary':
        pos_label = positive_label if positive_label is not None else classes[1]
        pos_idx = list(classes).index(pos_label)
        y_prob = model.predict_proba(X_val)[:, pos_idx]
        y_true_bin = (y_val == pos_label).astype(int)
        roc = roc_auc_score(y_true_bin, y_prob)
        pr = average_precision_score(y_true_bin, y_prob)
        brier = brier_score_loss(y_true_bin, y_prob)
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        y_pred = (y_prob >= best_thresh).astype(int)
        bal_acc = balanced_accuracy_score(y_true_bin, y_pred)
        for metric, value in [('roc_auc', roc), ('pr_auc', pr), ('balanced_accuracy', bal_acc), ('brier_score', brier)]:
            rows.append({'model': name, 'metric': metric, 'value': value, 'threshold': best_thresh})
        mlca_runtime.record_metric(primary_metric, roc, split='val')
    else:
        y_prob = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        roc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro', labels=classes)
        f1 = f1_score(y_val, y_pred, average='macro')
        for metric, value in [('roc_auc_ovr', roc), ('f1_macro', f1)]:
            rows.append({'model': name, 'metric': metric, 'value': value, 'threshold': np.nan})
        mlca_runtime.record_metric(primary_metric, roc, split='val')

df = pd.DataFrame(rows)
df['value'] = df['value'].map(lambda x: float(f'{x:.4g}'))
df['threshold'] = df['threshold'].map(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics for candidate models')

if task_type == 'binary':
    best_row = df[df['metric'] == 'roc_auc'].sort_values('value', ascending=False).iloc[0]
else:
    best_row = df[df['metric'] == 'roc_auc_ovr'].sort_values('value', ascending=False).iloc[0]
mlca_runtime.record_fact_candidate('best_model', best_row['model'], source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_row['value'], source='validation_metrics.csv')
