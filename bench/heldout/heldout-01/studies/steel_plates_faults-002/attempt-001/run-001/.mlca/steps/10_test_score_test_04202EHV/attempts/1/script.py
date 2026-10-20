import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
test = pd.read_parquet(inputs['test'])

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    nm = str(c).lower()
    if ('id' in nm or 'code' in nm) and train[c].nunique(dropna=True) == len(train):
        continue
    feat_cols.append(c)

# Validation-only model selection (freeze before any test exposure)
cand_paths = {
    'baseline': inputs['07_model_train_baseline_2DXYNCH8/model.pkl'],
    'alternative': inputs['08_model_train_alternative_QF453D1M/alternative_model.pkl'],
}
model_path, model_name = inputs['latest_model'], 'selected'
try:
    vm = pd.read_csv(inputs['09_validate_score_validation_FQ87WBM2/validation_metrics.csv'])
    if {'model', 'metric', 'value'}.issubset(vm.columns):
        sub = vm[vm['metric'].astype(str) == primary_metric]
        if len(sub):
            best = sub.loc[sub['value'].idxmax()]
            bn = str(best['model']).lower()
            for k, p in cand_paths.items():
                if k in bn:
                    model_path, model_name = p, k
                    break
except Exception:
    pass

with open(model_path, 'rb') as f:
    model = pickle.load(f)

used = [c for c in getattr(model, 'feature_names_in_', feat_cols) if c in test.columns]
X_test = test[list(used)].copy()
y_pred = np.asarray(model.predict(X_test))
y_true = test[targets].values

def r(x):
    return float(f'{float(x):.4g}')

rows, per_f1, per_hl = [], [], []
for i, t in enumerate(targets):
    f1 = float(f1_score(y_true[:, i], y_pred[:, i], average='macro'))
    hl = float(hamming_loss(y_true[:, i], y_pred[:, i]))
    per_f1.append(f1); per_hl.append(hl)
    rows.append({'model': model_name, 'metric': f'{t}:macro_f1', 'value': r(f1)})
    rows.append({'model': model_name, 'metric': f'{t}:hamming_loss', 'value': r(hl)})

macro_f1 = float(np.mean(per_f1))
avg_hl = float(np.mean(per_hl))
subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))
rows.append({'model': model_name, 'metric': 'macro_f1', 'value': r(macro_f1)})
rows.append({'model': model_name, 'metric': 'hamming_loss', 'value': r(avg_hl)})
rows.append({'model': model_name, 'metric': 'subset_accuracy', 'value': r(subset_acc)})

mlca_runtime.save_table(pd.DataFrame(rows), 'test_metrics.csv',
                        caption='Test metrics for declared multilabel targets')
mlca_runtime.record_metric(primary_metric, macro_f1, 'test')
mlca_runtime.record_fact_candidate('test_score', macro_f1, source='test_metrics.csv')
print('features', len(used), 'test_rows', len(test))
print(primary_metric, round(macro_f1, 4))
