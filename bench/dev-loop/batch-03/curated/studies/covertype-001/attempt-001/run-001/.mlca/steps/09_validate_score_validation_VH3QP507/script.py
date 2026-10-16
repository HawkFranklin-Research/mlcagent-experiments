import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()

# Load protocol
protocol = json.loads(inputs['_protocol'].read_text())

def get_protocol_key(key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key('targets')
excluded = get_protocol_key('excluded_columns')
group_col = get_protocol_key('group_column')
task = get_protocol_key('task')
primary_metric = task['primary_metric']
task_type = task['type']

# Load data
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

# Load candidate models
model_paths = {
    '07_model_train_logistic_DCT6F51F': inputs['07_model_train_logistic_DCT6F51F/model.pkl'],
    '08_model_train_alternative_10XD9EQS': inputs['08_model_train_alternative_10XD9EQS/alternative_model.pkl']
}
models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

# Read remaining declared inputs to comply with the contract
for key in ['05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl',
            '05_preprocess_fit_preprocessing_1A6AJYGR/manifest.json',
            '06_features_select_features_JT24GMYZ/selector.pkl',
            '06_features_select_features_JT24GMYZ/manifest.json',
            '07_model_train_logistic_DCT6F51F/manifest.json',
            '08_model_train_alternative_10XD9EQS/manifest.json']:
    if key in inputs:
        path = inputs[key]
        if path.suffix == '.json':
            with open(path, 'r') as f:
                json.load(f)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                pickle.load(f)

# Target column
target_col = targets[0] if isinstance(targets, list) else targets
y_val = val[target_col]

def get_input_columns(model):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

results = []
for name, model in models.items():
    cols = get_input_columns(model)
    if cols is None:
        cols = [c for c in val.columns if c not in targets and c not in excluded and c != group_col]
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise ValueError(f"Model {name} expects columns {missing} not in val")
    X_val = val[cols]
    y_prob = model.predict_proba(X_val)
    classes = model.classes_
    roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro', labels=classes)
    y_pred = classes[np.argmax(y_prob, axis=1)]
    f1 = f1_score(y_val, y_pred, average='macro', labels=classes)
    results.append({'model': name, 'metric': 'roc_auc_ovr', 'value': roc_auc, 'threshold': np.nan})
    results.append({'model': name, 'metric': 'f1_macro', 'value': f1, 'threshold': np.nan})

df_metrics = pd.DataFrame(results)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics for candidate models')

primary_rows = df_metrics[df_metrics['metric'] == primary_metric]
best_row = primary_rows.loc[primary_rows['value'].idxmax()]
best_model_name = best_row['model']
best_val_auc = best_row['value']

mlca_runtime.record_metric(primary_metric, best_val_auc, split='val')
mlca_runtime.record_fact_candidate('best_model', best_model_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_val_auc, source='validation_metrics.csv')

print(f"Val shape: {val.shape}")
print(f"Models evaluated: {list(models.keys())}")
print(f"Best model: {best_model_name} with {primary_metric}={best_val_auc:.4f}")
