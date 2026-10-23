import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
primary = protocol['task']['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)

feats = list(model.feature_names_in_)
test = pd.read_parquet(inputs['test'])
missing = [c for c in feats if c not in test.columns]
if missing:
    raise KeyError(f"test missing feature columns: {missing[:5]}")
X = test[feats]
target = targets[0]
y = test[target]

classes = list(model.classes_)
pos_label = classes[1]
pos_idx = classes.index(pos_label)
proba = model.predict_proba(X)
pp = proba[:, pos_idx]
yb = (y == pos_label).to_numpy().astype(int)

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if 'threshold' in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None: return r
    elif isinstance(o, list):
        for x in o:
            r = find_thr(x)
            if r is not None: return r
    return None

threshold = None
try:
    vman = json.loads(inputs['09_validate_score_validation_5DNAF4V8/manifest.json'].read_text())
    threshold = find_thr(vman)
except Exception:
    pass
if threshold is None:
    val = pd.read_parquet(inputs['val'])
    pv = model.predict_proba(val[feats])[:, pos_idx]
    yv = (val[target] == pos_label).to_numpy().astype(int)
    fpr, tpr, thr = roc_curve(yv, pv)
    threshold = float(thr[np.argmax(tpr - fpr)])

yp = (pp >= threshold).astype(int)

def metrics(yb, pp, yp):
    r = roc_auc_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
    p = average_precision_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
    ba = balanced_accuracy_score(yb, yp)
    br = float(np.mean((yb.astype(float) - pp) ** 2))
    tp = int(((yp == 1) & (yb == 1)).sum()); fn = int(((yp == 0) & (yb == 1)).sum())
    tn = int(((yp == 0) & (yb == 0)).sum()); fp = int(((yp == 1) & (yb == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {'roc_auc': r, 'pr_auc': p, 'balanced_accuracy': ba, 'brier': br,
            'sensitivity': sens, 'specificity': spec}

point = metrics(yb, pp, yp)
names = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier', 'sensitivity', 'specificity']
rng = np.random.RandomState(42)
n = len(yb)
boots = {k: [] for k in names}
for _ in range(1000):
    i = rng.randint(0, n, n)
    m = metrics(yb[i], pp[i], yp[i])
    for k in names:
        boots[k].append(m[k])

rows = []
for k in names:
    bs = np.array(boots[k], dtype=float); bs = bs[~np.isnan(bs)]
    lo = float(np.percentile(bs, 2.5)) if bs.size else np.nan
    hi = float(np.percentile(bs, 97.5)) if bs.size else np.nan
    rows.append({'metric': k, 'value': point[k], 'ci_low': lo, 'ci_high': hi})
mdf = pd.DataFrame(rows)
mlca_runtime.save_table(mdf, 'test_metrics.csv', caption='Test metrics, 95% bootstrap CI (1000 resamples)')

prow = mdf[mdf['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prow['value']), 'test',
                           ci=(float(prow['ci_low']), float(prow['ci_high'])))
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(prow['value']), source='test_metrics.csv')

preds = pd.DataFrame({'row_id': test.index, 'partition': 'test',
                      f'y_true__{target}': y.to_numpy(),
                      f'y_pred__{target}': model.predict(X)})
for j, c in enumerate(classes):
    preds[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(preds, 'test_predictions.parquet')

print(f"test n={len(test)} feats={len(feats)} thr={threshold:.4f}")
print(f"{primary}={point[primary]:.4f} ci=[{prow['ci_low']:.4f},{prow['ci_high']:.4f}]")
