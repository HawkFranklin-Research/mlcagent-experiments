import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']
primary = str(task['primary_metric'])
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group = protocol.get('group_column')
target = targets[0]
train = pd.read_parquet(inp['train'])
test = pd.read_parquet(inp['test'])

vm = pd.read_parquet(inp['09_validate_score_validation_KTKYAT6K/validation_metrics.parquet'])
vm.columns = [str(c).lower() for c in vm.columns]
if 'metric' in vm.columns:
    vm['metric_norm'] = vm['metric'].astype(str).str.lower().str.replace(' ', '_').str.replace('-', '_')

def best_from_vm(metric):
    m = metric.lower().replace(' ', '_').replace('-', '_')
    if 'metric_norm' in vm.columns and 'value' in vm.columns:
        sub = vm[vm['metric_norm'] == m]
        if len(sub) and 'model' in sub.columns:
            return sub.groupby('model')['value'].mean().idxmax()
    if m in vm.columns:
        return vm[m].idxmax()
    return None

best_name = best_from_vm(primary)
keys = list(inp.keys())
def pick_key(name):
    if name is not None:
        s = str(name).lower()
        for k in keys:
            if (s in k.lower() or str(name) in k) and (k.endswith('.pkl') or k == 'latest_model'):
                return k
        if 'logistic' in s or '07' in s:
            return '07_model_train_logistic_ZJV6JEEV/model.pkl'
        if 'alternative' in s or '08' in s:
            return '08_model_train_alternative_YVXTQSN7/alternative_model.pkl'
    return 'latest_model' if 'latest_model' in inp else '07_model_train_logistic_ZJV6JEEV/model.pkl'

selected_key = pick_key(best_name)
model = None
for key in [selected_key, 'latest_model', '07_model_train_logistic_ZJV6JEEV/model.pkl', '08_model_train_alternative_YVXTQSN7/alternative_model.pkl']:
    if key in inp:
        try:
            with open(inp[key], 'rb') as f:
                m = pickle.load(f)
            if hasattr(m, 'predict_proba'):
                model, selected_key = m, key
                break
        except Exception:
            pass
if model is None:
    raise RuntimeError('No frozen model with predict_proba found')

drop = set(targets) | set(excluded) | ({group} if group else set())
if hasattr(model, 'feature_names_in_'):
    feature_cols = [c for c in list(model.feature_names_in_) if c not in drop]
    if len(feature_cols) != len(list(model.feature_names_in_)):
        raise RuntimeError('Model expects target/excluded columns')
else:
    feature_cols = [c for c in train.columns if c not in drop]

X_test = test[feature_cols].copy()
proba = np.asarray(model.predict_proba(X_test))
def get_classes(m):
    if hasattr(m, 'classes_'):
        return np.asarray(m.classes_)
    if hasattr(m, 'steps'):
        return np.asarray(m.steps[-1][1].classes_)
    return None
classes = get_classes(model)
if classes is None:
    classes = np.sort(train[target].unique())
pos = task.get('positive_label', classes[1] if len(classes) > 1 else classes[0])
p = proba[:, int(np.where(classes == pos)[0][0])]
y = test[target].to_numpy()

threshold = None
if 'metric_norm' in vm.columns and 'value' in vm.columns:
    tr = vm[vm['metric_norm'].str.contains('threshold', na=False)]
    if len(tr):
        threshold = float(tr['value'].iloc[0])
if threshold is None:
    try:
        man = json.loads(inp['09_validate_score_validation_KTKYAT6K/manifest.json'].read_text())
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
        threshold = find_thr(man)
    except Exception:
        pass
if threshold is None:
    threshold = 0.5

def safe(fn):
    try:
        return float(fn())
    except Exception:
        return np.nan

def calc(y_true, prob):
    yb = (y_true == pos).astype(int)
    yhat = (prob >= threshold).astype(int)
    return {
        'roc_auc': safe(lambda: roc_auc_score(yb, prob)),
        'pr_auc': safe(lambda: average_precision_score(yb, prob)),
        'balanced_accuracy': safe(lambda: balanced_accuracy_score(yb, yhat)),
        'brier_score': safe(lambda: np.mean((prob - yb) ** 2)),
        'sensitivity': safe(lambda: ((yhat == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)),
        'specificity': safe(lambda: ((yhat == 0) & (yb == 0)).sum() / max((yb == 0).sum(), 1)),
    }

point = calc(y, p)
metric_keys = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score', 'sensitivity', 'specificity']
rng = np.random.default_rng(42)
boot = {k: [] for k in metric_keys}
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    m = calc(y[idx], p[idx])
    for k in metric_keys:
        boot[k].append(m[k])

rows = []
for k in metric_keys:
    vals = np.asarray(boot[k], dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    rows.append({'metric': primary if k.lower() == primary.lower() else k,
                 'value': point[k], 'ci_low': lo, 'ci_high': hi})
out_df = pd.DataFrame(rows)
mlca_runtime.save_table(out_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')
for _, r in out_df.iterrows():
    mlca_runtime.record_metric(str(r['metric']), float(r['value']), 'test',
                               ci=(float(r['ci_low']), float(r['ci_high'])))
if primary.lower() == 'roc_auc':
    r = out_df[out_df['metric'] == primary].iloc[0]
    mlca_runtime.record_fact_candidate('test_roc_auc', float(r['value']), source='test_metrics.csv')
print(f'selected={selected_key} n_test={len(test)} n_features={len(feature_cols)} threshold={threshold:.4g}')
pr = out_df[out_df['metric'] == primary].iloc[0]
print(f'{primary}={pr["value"]:.4g} CI=({pr["ci_low"]:.4g},{pr["ci_high"]:.4g})')
