import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group_col = proto['group_column']
pm = proto['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

exclude_set = set(targets) | set(excluded) | {group_col}
feature_cols = [c for c in train.columns if c not in exclude_set]

candidates = {
    'baseline': inputs['07_model_train_baseline_N5M7BNX3/model.pkl'],
    'alternative': inputs['08_model_train_alternative_A6RD3R24/alternative_model.pkl'],
}
val_scores = {}
for name, path in candidates.items():
    m = pickle.load(open(path, 'rb'))
    fc = list(m.feature_names_in_) if hasattr(m, 'feature_names_in_') else feature_cols
    pv = m.predict(val[fc])
    yv = val[targets].values
    f1s = [float(f1_score(np.asarray(yv[:, i]), np.asarray(pv[:, i]), average='macro'))
           for i in range(len(targets))]
    val_scores[name] = float(np.mean(f1s))

best = max(val_scores, key=val_scores.get)
print(f"Selected {best} via val {pm}={val_scores[best]:.4f}")

model = pickle.load(open(candidates[best], 'rb'))
fc = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feature_cols
y_true = np.asarray(test[targets].values)
y_pred = np.asarray(model.predict(test[fc]))

rows = []
f1_vals, hl_vals = [], []
for i, t in enumerate(targets):
    f1 = float(f1_score(y_true[:, i], y_pred[:, i], average='macro'))
    hl = float(hamming_loss(y_true[:, i], y_pred[:, i]))
    f1_vals.append(f1); hl_vals.append(hl)
    rows.append({'model': best, 'metric': f'macro_f1_{t}', 'value': f1})
    rows.append({'model': best, 'metric': f'hamming_loss_{t}', 'value': hl})

subset_acc = float(np.mean([bool(np.all(y_true[i] == y_pred[i])) for i in range(len(y_true))]))
f1_avg = float(np.mean(f1_vals))
hl_avg = float(np.mean(hl_vals))
rows.append({'model': best, 'metric': 'macro_f1', 'value': f1_avg})
rows.append({'model': best, 'metric': 'hamming_loss', 'value': hl_avg})
rows.append({'model': best, 'metric': 'subset_accuracy', 'value': subset_acc})

mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv")
mlca_runtime.record_metric(pm, f1_avg, "test")
mlca_runtime.record_fact_candidate("test_score", f1_avg, source="test_metrics.csv")
print(f"Test {pm}={f1_avg:.4f} subset_acc={subset_acc:.4f} n={len(y_true)}")
