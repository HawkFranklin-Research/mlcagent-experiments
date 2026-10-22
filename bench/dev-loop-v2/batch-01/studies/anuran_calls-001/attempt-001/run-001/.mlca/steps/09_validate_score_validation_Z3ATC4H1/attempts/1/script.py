import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

val = pd.read_parquet(inputs['val'])

id_like = []
for c in val.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = val[c]
    if s.dtype == object or str(s.dtype).startswith('string') or pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            id_like.append(c)
feature_columns = [c for c in val.columns
                   if c not in targets and c not in excluded
                   and c != group_col and c not in id_like]

candidates = {
    'baseline': inputs['07_model_train_baseline_N5M7BNX3/model.pkl'],
    'alternative': inputs['08_model_train_alternative_A6RD3R24/alternative_model.pkl'],
}

def score(vals, preds):
    yt = vals[targets].values
    pr = preds if preds.ndim == 2 else preds.reshape(-1, len(targets))
    per_f1, per_hl = [], []
    for i, t in enumerate(targets):
        a, b = yt[:, i], pr[:, i]
        if a.dtype == object or b.dtype == object:
            a, b = a.astype(str), b.astype(str)
        per_f1.append(float(f1_score(a, b, average='macro', zero_division=0)))
        per_hl.append(float(hamming_loss(a, b)))
    a, b = yt.ravel(), pr.ravel()
    if a.dtype == object or b.dtype == object:
        a, b = a.astype(str), b.astype(str)
    sub = float(np.mean(np.all(b.reshape(pr.shape) == a.reshape(yt.shape), axis=1)))
    return {'macro_f1': float(np.mean(per_f1)),
            'hamming_loss': float(np.mean(per_hl)),
            'subset_accuracy': sub}

results = {}
for name, path in candidates.items():
    with open(path, 'rb') as f:
        model = pickle.load(f)
    fn = getattr(model, 'feature_names_in_', None)
    cols = list(fn) if fn is not None else feature_columns
    preds = np.asarray(model.predict(val[cols]))
    results[name] = score(val, preds)

best_name = max(results, key=lambda n: results[n]['macro_f1'])
best = results[best_name]

rows = []
for name, res in results.items():
    for met in ['macro_f1', 'hamming_loss', 'subset_accuracy']:
        rows.append({'model': name, 'metric': met, 'value': round(res[met], 4)})
mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv')

mlca_runtime.record_metric(primary_metric, best['macro_f1'], 'val')
mlca_runtime.record_fact_candidate('best_validation_score', float(best['macro_f1']),
                                   unit=primary_metric, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')

print('val_shape', val.shape)
print('best_model', best_name)
print(primary_metric, round(best['macro_f1'], 4))
