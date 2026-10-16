import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

val_metrics = pd.read_csv(inputs['09_validate_score_validation_DGS5S5Y8/validation_metrics.csv'])
cand = val_metrics[val_metrics['metric'] == primary_metric]
if len(cand) == 0:
    cand = val_metrics[val_metrics['metric'].str.contains(primary_metric, na=False)]
best_row = cand.sort_values('value', ascending=False).iloc[0]
best_model_name = str(best_row['model'])

if 'baseline' in best_model_name.lower():
    model_path = inputs['07_model_train_baseline_SGXK026F/model.pkl']
elif 'alternative' in best_model_name.lower():
    model_path = inputs['08_model_train_alternative_MK5K74KS/alternative_model.pkl']
else:
    model_path = inputs['latest_model']

with open(model_path, 'rb') as f:
    model = pickle.load(f)

train = pd.read_parquet(inputs['train'])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

test = pd.read_parquet(inputs['test'])
X_test = test[feature_cols]
y_test = test[targets]

preds = model.predict(X_test)
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

rows = []
f1s = []
hl_list = []
for i, t in enumerate(targets):
    f1 = float(f1_score(y_test[t].values, preds[:, i], average='macro'))
    f1s.append(f1)
    rows.append({'model': best_model_name, 'metric': f'macro_f1_{t}', 'value': f1})
    hl = float(hamming_loss(y_test[t].values, preds[:, i]))
    hl_list.append(hl)

mean_f1 = float(np.mean(f1s))
mean_hl = float(np.mean(hl_list))
subset_acc = float(np.mean(np.all(preds == y_test.values, axis=1)))

rows.append({'model': best_model_name, 'metric': 'macro_f1', 'value': mean_f1})
rows.append({'model': best_model_name, 'metric': 'hamming_loss', 'value': mean_hl})
rows.append({'model': best_model_name, 'metric': 'subset_accuracy', 'value': subset_acc})

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics')

mlca_runtime.record_metric(primary_metric, mean_f1, 'test')
mlca_runtime.record_fact_candidate('test_score', mean_f1, source='test_metrics.csv')

print(f"test rows: {len(test)}")
print(f"best model: {best_model_name}")
print(f"{primary_metric}: {mean_f1:.4f}")
