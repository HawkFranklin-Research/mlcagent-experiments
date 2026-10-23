import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary = task['primary_metric']
targets = list(protocol['targets'])
if len(targets) != 1:
    raise ValueError('expected exactly one target')
target = targets[0]
if 'excluded_columns' not in protocol:
    raise KeyError(f"excluded_columns missing; top-level keys: {list(protocol.keys())}")
excluded = set(protocol['excluded_columns'] or [])
if 'group_column' not in protocol:
    raise KeyError(f"group_column missing; top-level keys: {list(protocol.keys())}")
group = protocol['group_column']
exclude = excluded | set(targets) | ({group} if group else set())

test = pd.read_parquet(inputs['test'])
model = pickle.load(open(inputs['latest_model'], 'rb'))

if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = []
    for c in test.columns:
        if c in exclude:
            continue
        s = test[c]
        is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
        if is_text and s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(c)

bad = set(feature_cols) & exclude
if bad:
    raise ValueError(f"feature list includes outcome/excluded columns: {bad}")
missing = [c for c in feature_cols if c not in test.columns]
if missing:
    raise KeyError(f"missing feature columns in test: {missing}")
X = test[feature_cols]

if hasattr(model, 'classes_'):
    classes = np.asarray(model.classes_)
else:
    last = model.steps[-1][1] if hasattr(model, 'steps') and model.steps else model
    classes = np.asarray(last.classes_)
if len(classes) != 2:
    raise ValueError('binary task but model does not have exactly two classes')

proba = model.predict_proba(X)
if proba.shape[1] != len(classes):
    raise ValueError('probability columns do not match model classes')
pos_label = task.get('positive_label', protocol.get('positive_label'))
pos_idx = 1 if pos_label is None else list(classes).index(pos_label)
pos_label = classes[pos_idx]
neg_label = classes[1 - pos_idx]
p_pos = proba[:, pos_idx]

def find_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'threshold' in str(k).lower():
                try:
                    return float(v)
                except Exception:
                    pass
            r = find_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = find_threshold(v)
            if r is not None:
                return r
    return None

def threshold_from_df(df):
    for c in df.columns:
        if 'threshold' in str(c).lower():
            v = pd.to_numeric(df[c], errors='coerce').dropna()
            if len(v):
                return float(v.iloc[0])
    mcol = next((c for c in df.columns if str(c).lower() in ('metric', 'name', 'key')), None)
    vcol = next((c for c in df.columns if str(c).lower() in ('value', 'score')), None)
    if mcol and vcol:
        m = df[mcol].astype(str).str.lower().str.contains('threshold')
        v = pd.to_numeric(df.loc[m, vcol], errors='coerce').dropna()
        if len(v):
            return float(v.iloc[0])
    return None

threshold = None
for key in ['10_validate_score_validation_0582SCTN/validation_metrics.csv',
            '10_validate_score_validation_0582SCTN/validation_metrics.parquet']:
    if threshold is not None:
        break
    if key.endswith('.csv'):
        threshold = threshold_from_df(pd.read_csv(inputs[key]))
    else:
        threshold = threshold_from_df(pd.read_parquet(inputs[key]))
if threshold is None:
    threshold = find_threshold(json.loads(inputs['10_validate_score_validation_0582SCTN/manifest.json'].read_text()))
if threshold is None:
    threshold = 0.5

y_true = test[target].to_numpy()
y_bin = (y_true == pos_label).astype(int)
pred_bin = (p_pos >= threshold).astype(int)
y_pred = np.where(pred_bin == 1, pos_label, neg_label)

def calc_metrics(y, yb, p, pb):
    out = {'roc_auc': np.nan, 'pr_auc': np.nan, 'balanced_accuracy': np.nan,
           'brier_score': brier_score_loss(yb, p), 'sensitivity': np.nan, 'specificity': np.nan}
    if len(np.unique(yb)) == 2:
        out['roc_auc'] = roc_auc_score(yb, p)
        out['pr_auc'] = average_precision_score(yb, p)
    tp = int(np.sum((yb == 1) & (pb == 1)))
    fn = int(np.sum((yb == 1) & (pb == 0)))
    tn = int(np.sum((yb == 0) & (pb == 0)))
    fp = int(np.sum((yb == 0) & (pb == 1)))
    if tp + fn:
        out['sensitivity'] = tp / (tp + fn)
    if tn + fp:
        out['specificity'] = tn / (tn + fp)
    if np.isfinite(out['sensitivity']) and np.isfinite(out['specificity']):
        out['balanced_accuracy'] = (out['sensitivity'] + out['specificity']) / 2
    return out

metrics = calc_metrics(y_true, y_bin, p_pos, pred_bin)
names = [primary] + [k for k in metrics if k != primary]
rng = np.random.default_rng(SEED)
boot = {k: [] for k in metrics}
n = len(y_true)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    m = calc_metrics(y_true[idx], y_bin[idx], p_pos[idx], pred_bin[idx])
    for k, v in m.items():
        if np.isfinite(v):
            boot[k].append(v)

ci = {}
for k in metrics:
    arr = np.asarray(boot[k])
    ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

def sig4(x):
    return np.nan if not np.isfinite(x) else float(f"{x:.4g}")

rows = []
for k in names:
    lo, hi = ci[k]
    rows.append({'metric': k, 'value': sig4(metrics[k]), 'ci_low': sig4(lo), 'ci_high': sig4(hi)})
mlca_runtime.save_table(pd.DataFrame(rows), 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

for k in names:
    mlca_runtime.record_metric(k, float(metrics[k]), 'test', ci=ci[k])
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(metrics['roc_auc']), source='test_metrics.csv')

pred_df = pd.DataFrame({'row_id': test.index, 'partition': 'test', 'y_true': y_true, 'y_pred': y_pred})
for j, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"rows={len(test)} features={len(feature_cols)} threshold={threshold:g} pos={pos_label}")
print(f"{primary}={metrics[primary]:.4g} ci=({ci[primary][0]:.4g},{ci[primary][1]:.4g})")
