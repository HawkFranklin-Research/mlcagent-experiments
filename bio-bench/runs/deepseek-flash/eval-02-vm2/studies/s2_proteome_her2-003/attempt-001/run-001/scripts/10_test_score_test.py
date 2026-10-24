import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp['_protocol'].read_text())
primary = prot['task']['primary_metric']
target = prot['targets'][0]

model = pickle.loads(inp['latest_model'].read_bytes())
classes = list(model.classes_)
feat_cols = list(model.feature_names_in_)

test = pd.read_parquet(inp['test'])
Xt = test[feat_cols]
proba = model.predict_proba(Xt)
y_true = test[target].to_numpy()
pos_label = classes[1]
p1 = proba[:, classes.index(pos_label)]
yb = (y_true == pos_label).astype(int)

thr = 0.5
try:
    man = json.loads(inp['09_validate_score_validation_NVJWW7P1/manifest.json'].read_text())
    def find_thr(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if 'thresh' in k.lower() and isinstance(v, (int, float)):
                    return float(v)
                r = find_thr(v)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for it in d:
                r = find_thr(it)
                if r is not None:
                    return r
        return None
    t = find_thr(man)
    if t is not None:
        thr = t
except Exception:
    pass
yhat = np.where(p1 >= thr, pos_label, classes[0])

def compute(idx):
    pp, bb, yy, hh = p1[idx], yb[idx], y_true[idx], yhat[idx]
    o = {}
    o['roc_auc'] = roc_auc_score(bb, pp) if len(np.unique(bb)) > 1 else np.nan
    o['pr_auc'] = average_precision_score(bb, pp) if len(np.unique(bb)) > 1 else np.nan
    o['balanced_accuracy'] = balanced_accuracy_score(yy, hh)
    o['brier_score'] = brier_score_loss(bb, pp)
    tp = int(((hh == pos_label) & (yy == pos_label)).sum())
    fn = int(((hh != pos_label) & (yy == pos_label)).sum())
    tn = int(((hh != pos_label) & (yy != pos_label)).sum())
    fp = int(((hh == pos_label) & (yy != pos_label)).sum())
    o['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    o['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return o

idx0 = np.arange(len(y_true))
point = compute(idx0)
keys = list(point.keys())
rng = np.random.default_rng(42)
boots = {k: [] for k in keys}
for _ in range(1000):
    bi = rng.integers(0, len(y_true), len(y_true))
    try:
        bm = compute(bi)
        for k in keys:
            boots[k].append(bm[k])
    except Exception:
        pass

rows = []
for k in keys:
    arr = np.array([x for x in boots[k] if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    lo = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
    hi = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    rows.append({'metric': k, 'value': point[k], 'ci_low': lo, 'ci_high': hi})
tm = pd.DataFrame(rows)
mlca_runtime.save_table(tm.round(4), 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

pv = float(point[primary])
row = tm[tm.metric == primary].iloc[0]
mlca_runtime.record_metric(primary, pv, 'test', ci=[float(row.ci_low), float(row.ci_high)])
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', pv, source='test_metrics.csv')

pred = pd.DataFrame({
    'row_id': test.index,
    'partition': 'test',
    'y_true__' + target: y_true,
    'y_pred__' + target: yhat,
})
for j, c in enumerate(classes):
    pred['proba__' + str(c)] = proba[:, j]
mlca_runtime.save_row_outputs(pred, 'test_predictions.parquet')
print('test rows:', len(pred), 'primary', primary, '=', round(pv, 4), 'thr', round(thr, 4))
