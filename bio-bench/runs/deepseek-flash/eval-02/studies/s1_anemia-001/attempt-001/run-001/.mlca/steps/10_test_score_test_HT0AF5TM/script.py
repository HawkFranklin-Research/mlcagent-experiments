import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ('task', 'targets'):
    if k not in proto:
        raise KeyError(f"missing {k!r}; available top-level keys: {list(proto)}")
tm = proto['task']['primary_metric']
target = proto['targets'][0]

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if 'threshold' in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_thr(v)
            if r is not None:
                return r
    return None

thr = None
for k, v in inp.items():
    if k.endswith('manifest.json'):
        try:
            t = find_thr(json.loads(v.read_text()))
        except Exception:
            t = None
        if t is not None:
            thr = t
            break
if thr is None:
    thr = 0.5

with open(inp['latest_model'], 'rb') as f:
    model = pickle.load(f)
test = pd.read_parquet(inp['test'])
feats = list(model.feature_names_in_)
X = test[feats]
ytrue = test[target].to_numpy()
proba = model.predict_proba(X)
classes = list(model.classes_)
pos = proto['task'].get('positive_label', classes[1])
pos_idx = classes.index(pos)
other = classes[1 - pos_idx]
ind = (ytrue == pos).astype(int)
p = proba[:, pos_idx]
ypred = np.where(p >= thr, pos, other)

def metrics(ind, p, yv, ypr, pos):
    tp = int(((ypr == pos) & (yv == pos)).sum()); fn = int(((ypr != pos) & (yv == pos)).sum())
    tn = int(((ypr != pos) & (yv != pos)).sum()); fp = int(((ypr == pos) & (yv != pos)).sum())
    d = {'balanced_accuracy': balanced_accuracy_score(yv, ypr),
         'brier_score': brier_score_loss(ind, p),
         'sensitivity': tp / (tp + fn) if tp + fn else np.nan,
         'specificity': tn / (tn + fp) if tn + fp else np.nan}
    if len(np.unique(ind)) == 2:
        d['roc_auc'] = roc_auc_score(ind, p)
        d['pr_auc'] = average_precision_score(ind, p)
    else:
        d['roc_auc'] = np.nan; d['pr_auc'] = np.nan
    return d

keys = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score', 'sensitivity', 'specificity']
point = metrics(ind, p, ytrue, ypred, pos)
rng = np.random.default_rng(42)
n = len(test)
boot = {k: [] for k in keys}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    d = metrics(ind[idx], p[idx], ytrue[idx], ypred[idx], pos)
    for k in keys:
        boot[k].append(d[k])

rows = []
for k in keys:
    arr = np.asarray(boot[k], dtype=float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    rows.append({'metric': k, 'value': round(float(point[k]), 4),
                 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4)})
res = pd.DataFrame(rows)
mlca_runtime.save_table(res, 'test_metrics.csv',
                        caption=f'Test metrics for frozen model (threshold={thr:.4f}, positive={pos}).')

pv = float(res.loc[res['metric'] == tm, 'value'].iloc[0])
lo = float(res.loc[res['metric'] == tm, 'ci_low'].iloc[0])
hi = float(res.loc[res['metric'] == tm, 'ci_high'].iloc[0])
mlca_runtime.record_metric(tm, pv, 'test', ci=(lo, hi))
if tm == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', pv, source='test_metrics.csv')

pred_df = pd.DataFrame({'row_id': test.index, 'partition': 'test',
                        'y_true': ytrue, 'y_pred': ypred})
for j, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')
print(f"n_test={n} primary={tm}={pv:.4f} ci=({lo:.4f},{hi:.4f}) thr={thr:.4f}")
