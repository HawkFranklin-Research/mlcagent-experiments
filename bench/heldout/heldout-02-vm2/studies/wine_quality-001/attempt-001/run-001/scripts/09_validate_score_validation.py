import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
primary = protocol['task']['primary_metric']
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_keylike(s):
    if s.dtype == object:
        return s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique() == len(s)
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group
                and not is_keylike(train[c])]

y_val = val[target].values
models = {
    'logistic': pickle.loads(inputs['07_model_train_logistic_AP2AWK8D/model.pkl'].read_bytes()),
    'alternative': pickle.loads(inputs['08_model_train_alternative_D8BF01JS/alternative_model.pkl'].read_bytes()),
}

rows, val_scores = [], {}
for name, model in models.items():
    feat = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feature_cols
    X = val[feat]
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    aucs = [roc_auc_score((y_val == c).astype(int), proba[:, j])
            for j, c in enumerate(classes) if c in np.unique(y_val)]
    roc_auc_ovr = float(np.mean(aucs))
    pred = classes[proba.argmax(axis=1)]
    rows.append({'model': name, 'metric': primary, 'value': roc_auc_ovr, 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'macro_f1', 'value': f1_score(y_val, pred, average='macro'), 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'balanced_accuracy', 'value': balanced_accuracy_score(y_val, pred), 'threshold': np.nan})
    val_scores[name] = roc_auc_ovr

best_model = max(val_scores, key=lambda k: val_scores[k])
mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv')
mlca_runtime.record_metric(primary, float(val_scores[best_model]), 'val')
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(val_scores[best_model]), source='validation_metrics.csv')
print(f"features={len(feature_cols)} val={len(val)} best={best_model} val_auc={val_scores[best_model]:.4f}")
