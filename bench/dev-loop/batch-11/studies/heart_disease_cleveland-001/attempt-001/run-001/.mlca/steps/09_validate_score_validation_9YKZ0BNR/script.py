import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_69FZY9R2/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_TEGHQMDA/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["07_model_train_logistic_AGQDNKZV/model.pkl"], "rb") as f:
    model = pickle.load(f)
with open(inputs["08_model_train_alternative_C0PZ290P/alternative_model.pkl"], "rb") as f:
    alt_model = pickle.load(f)

target = protocol['targets'][0]
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
primary_metric = protocol['task']['primary_metric']

def get_features(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    elif hasattr(obj, 'named_steps'):
        for step in obj.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

def evaluate(model_obj, name):
    feats = get_features(model_obj)
    if feats is None:
        feats = [c for c in train.columns if c not in [target] + excluded and (group_col is None or c != group_col)]
    X_val = val[feats]
    y_val = val[target]
    if not hasattr(model_obj, 'predict_proba'):
        raise ValueError("Model has no predict_proba")
    proba = model_obj.predict_proba(X_val)
    classes = model_obj.classes_
    pos_idx = list(classes).index(1) if 1 in classes else 1
    y_prob = proba[:, pos_idx]

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    y_pred = (y_prob >= best_threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_val, y_pred)

    return [
        {'model': name, 'metric': primary_metric, 'value': roc_auc, 'threshold': np.nan},
        {'model': name, 'metric': 'pr_auc', 'value': pr_auc, 'threshold': np.nan},
        {'model': name, 'metric': 'balanced_accuracy', 'value': bal_acc, 'threshold': best_threshold},
        {'model': name, 'metric': 'brier', 'value': brier, 'threshold': np.nan},
    ]

rows = []
rows.extend(evaluate(model, 'logistic_AGQDNKZV'))
rows.extend(evaluate(alt_model, 'alternative_C0PZ290P'))

df_metrics = pd.DataFrame(rows)
df_metrics['value'] = df_metrics['value'].round(4)
df_metrics['threshold'] = df_metrics['threshold'].round(4)
mlca_runtime.save_table(df_metrics, "validation_metrics.csv", caption="Validation metrics for each model")

roc_rows = df_metrics[df_metrics['metric'] == primary_metric]
if len(roc_rows) == 0:
    raise ValueError("No rows for primary metric found")
best_row = roc_rows.loc[roc_rows['value'].idxmax()]
best_model = best_row['model']
best_val_roc_auc = best_row['value']

mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_val_roc_auc, source='validation_metrics.csv')

print(f"Validation rows: {len(val)}")
print(f"Best model: {best_model} with {primary_metric} = {best_val_roc_auc}")
