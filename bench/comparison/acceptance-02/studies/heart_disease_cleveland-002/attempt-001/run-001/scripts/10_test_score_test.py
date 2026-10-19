import mlca_runtime
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inp['train'])
test = pd.read_parquet(inp['test'])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]

val_m = pd.read_csv(inp['09_validate_score_validation_CWBDSN2A/validation_metrics.csv'])
sub = val_m[val_m['metric'] == primary_metric]
if len(sub) == 0:
    raise ValueError(f"No validation rows for metric {primary_metric}; got {val_m['metric'].unique()}")
best_name = str(sub.loc[sub['value'].idxmax(), 'model'])
key = 'baseline' if 'baseline' in best_name.lower() else 'alternative'
paths = {'baseline': inp['07_model_train_baseline_AHTSSAGQ/model.pkl'],
         'alternative': inp['08_model_train_alternative_AC8TT8JH/alternative_model.pkl']}

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

model = load(paths[key])
if not hasattr(model, 'named_steps'):
    pre = load(inp['05_preprocess_fit_preprocessing_5Z83A35F/preprocessor.pkl'])
    sel = load(inp['06_features_retain_features_Z9TAK0X7/selector_all.pkl'])
    steps = []
    pre_cols = getattr(pre, 'feature_names_in_', None)
    sel_cols = getattr(sel, 'feature_names_in_', None)
    raw = set(feature_cols)
    if sel_cols is not None and pre_cols is not None and set(sel_cols) <= set(pre_cols):
        steps = [('frontend', sel), ('model', model)]
    else:
        steps = [('pre', pre), ('sel', sel), ('model', model)]
    model = Pipeline(steps)

X_test = test[feature_cols].copy()
raw_pred = model.predict(X_test)
pred = np.asarray(raw_pred)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
y_true = test[targets].values

per_f1 = []
per_ham = []
for i, t in enumerate(targets):
    per_f1.append(float(f1_score(y_true[:, i], pred[:, i], average='macro')))
    per_ham.append(float(hamming_loss(y_true[:, i], pred[:, i])))

macro_f1 = float(np.mean(per_f1))
hamming = float(np.mean(per_ham))
subset_acc = float(np.mean(np.all(y_true == pred, axis=1)))

def rs(v):
    return float(f"{v:.4g}")

rows = [
    {'model': best_name, 'metric': 'macro_f1', 'value': rs(macro_f1)},
    {'model': best_name, 'metric': 'hamming_loss', 'value': rs(hamming)},
    {'model': best_name, 'metric': 'subset_accuracy', 'value': rs(subset_acc)},
]
for i, t in enumerate(targets):
    rows.append({'model': best_name, 'metric': f'macro_f1__{t}', 'value': rs(per_f1[i])})
    rows.append({'model': best_name, 'metric': f'hamming_loss__{t}', 'value': rs(per_ham[i])})

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption=f'Test metrics for {best_name}')
mlca_runtime.record_metric(primary_metric, macro_f1, 'test')
mlca_runtime.record_fact_candidate('test_score', macro_f1, source='test_metrics.csv')
print(f"selected={best_name} macro_f1={macro_f1:.4f} hamming={hamming:.4f} subset_acc={subset_acc:.4f}")
