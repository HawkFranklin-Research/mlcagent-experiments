import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task['type']
n_classes = task.get('n_classes', 2)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

model_paths = {
    'logistic': inputs['07_model_train_logistic_GZE7Y5E0/model.pkl'],
    'alternative': inputs['08_model_train_alternative_EJHSTA6Q/alternative_model.pkl']
}
models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

first_model = list(models.values())[0]
if hasattr(first_model, 'classes_'):
    classes = first_model.classes_
elif hasattr(first_model, 'named_steps'):
    classes = first_model.named_steps[list(first_model.named_steps.keys())[-1]].classes_
else:
    classes = None
positive_label = classes[1] if classes is not None and len(classes) == 2 else None
print(f"Positive label convention: {positive_label}")

y_val = val[targets[0]] if len(targets) == 1 else val[targets]
if isinstance(y_val, pd.DataFrame):
    y_val = y_val.iloc[:, 0]

metrics_rows = []
model_results = {}
for name, model in models.items():
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps'):
        first_step = model.named_steps[list(model.named_steps.keys())[0]]
        feature_cols = list(first_step.feature_names_in_) if hasattr(first_step, 'feature_names_in_') else [c for c in train.columns if c not in targets + excluded + ([group] if group else [])]
    else:
        feature_cols = [c for c in train.columns if c not in targets + excluded + ([group] if group else [])]
    X_val = val[feature_cols]
    proba = model.predict_proba(X_val)
    if hasattr(model, 'classes_'):
        model_classes = model.classes_
    elif hasattr(model, 'named_steps'):
        model_classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    else:
        model_classes = None
    if model_classes is not None and positive_label is not None and positive_label in model_classes:
        pos_idx = list(model_classes).index(positive_label)
    else:
        pos_idx = 1 if proba.shape[1] == 2 else 0
    y_prob = proba[:, pos_idx]
    y_true_bin = (y_val == positive_label).astype(int) if positive_label is not None else (y_val == model_classes[1]).astype(int)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    pr_auc = average_precision_score(y_true_bin, y_prob)
    brier = brier_score_loss(y_true_bin, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
    j_scores = tpr - fpr
    finite = np.isfinite(thresholds)
    best_threshold = thresholds[finite][np.argmax(j_scores[finite])] if finite.any() else 0.5
    y_pred_bin = (y_prob >= best_threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    model_results[name] = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'balanced_accuracy': bal_acc,
        'brier_score': brier,
        'threshold': best_threshold
    }
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score']:
        thr = float(f"{best_threshold:.4g}") if metric == 'balanced_accuracy' else np.nan
        metrics_rows.append({
            'model': name,
            'metric': metric,
            'value': float(f"{model_results[name][metric]:.4g}"),
            'threshold': thr
        })

best_model_name = max(model_results, key=lambda k: model_results[k][primary_metric])
best_val = model_results[best_model_name][primary_metric]

df_metrics = pd.DataFrame(metrics_rows)
mlca_runtime.save_table(df_metrics, "validation_metrics.csv", caption="Validation metrics for candidate models")

for name in models:
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score']:
        mlca_runtime.record_metric(f"{name}_{metric}", model_results[name][metric], split='val')
mlca_runtime.record_metric(primary_metric, best_val, split='val')
mlca_runtime.record_fact_candidate("best_model", best_model_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val, source="validation_metrics.csv")

print(f"Train shape: {train.shape}, Val shape: {val.shape}")
print(f"Best model: {best_model_name}, {primary_metric}={best_val:.4f}")
print(f"Threshold for best model: {model_results[best_model_name]['threshold']:.4f}")
