import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
I = mlca_runtime.inputs()
proto = json.loads(I['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group_col = proto.get('group_column', None)
task = proto['task']
primary = task['primary_metric']
higher_better = primary.lower() in ('r2', 'accuracy', 'auc', 'f1', 'precision', 'recall')

train = pd.read_parquet(I['train'])
val = pd.read_parquet(I['val'])
test = pd.read_parquet(I['test'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype.kind in ('O', 'U', 'S') and s.nunique() == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        continue
    feat_cols.append(c)

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(I['05_preprocess_fit_preprocessing_RHXAHV44/preprocessor.pkl'])
sel = load(I['06_features_retain_features_W82TE9KX/selector_all.pkl'])
m7 = load(I['07_model_train_baseline_1GT1TPB8/model.pkl'])
m8 = load(I['08_model_train_alternative_KVXQRP8Q/alternative_model.pkl'])

def first_accepts_raw(obj):
    if isinstance(obj, Pipeline):
        steps = list(obj.named_steps.values())
        if not steps:
            return False
        obj = steps[0]
    if hasattr(obj, 'feature_names_in_'):
        try:
            return set(obj.feature_names_in_) == set(feat_cols)
        except Exception:
            return False
    return False

def as_full(m):
    if isinstance(m, Pipeline):
        names = [n.lower() for n in m.named_steps]
        if any('pre' in n or 'select' in n or 'transform' in n for n in names):
            return m
    if first_accepts_raw(sel):
        return Pipeline([('sel', sel), ('model', m)])
    return Pipeline([('pre', pre), ('sel', sel), ('model', m)])

def mname(p, d):
    man = json.loads(p.read_text())
    for k in ['model_name', 'name', 'estimator_name', 'estimator']:
        if k in man and isinstance(man[k], str):
            return man[k]
    return d

name7 = mname(I['07_model_train_baseline_1GT1TPB8/manifest.json'], 'baseline')
name8 = mname(I['08_model_train_alternative_KVXQRP8Q/manifest.json'], 'alternative')

def primary_score(pipe, X, y):
    pred = pipe.predict(X)
    if pred.ndim == 1:
        pred = pred.reshape(-1, len(targets))
    vals = []
    for i, t in enumerate(targets):
        if primary.lower() == 'rmse':
            v = float(np.sqrt(mean_squared_error(y[t], pred[:, i])))
        elif primary.lower() == 'mae':
            v = float(mean_absolute_error(y[t], pred[:, i]))
        elif primary.lower() == 'r2':
            v = float(r2_score(y[t], pred[:, i]))
        else:
            v = float(r2_score(y[t], pred[:, i]))
        vals.append(v)
    return float(np.mean(vals))

pipes = {name7: as_full(m7), name8: as_full(m8)}
vs = {k: primary_score(v, val[feat_cols], val[targets]) for k, v in pipes.items()}
best = (max if higher_better else min)(vs, key=vs.get)
best_pipe = pipes[best]

pred = best_pipe.predict(test[feat_cols])
if pred.ndim == 1:
    pred = pred.reshape(-1, len(targets))
y = test[targets]
rows, macro = [], {}
for metric in ['rmse', 'mae', 'r2']:
    vals = []
    for i, t in enumerate(targets):
        if metric == 'rmse':
            v = float(np.sqrt(mean_squared_error(y[t], pred[:, i])))
        elif metric == 'mae':
            v = float(mean_absolute_error(y[t], pred[:, i]))
        else:
            v = float(r2_score(y[t], pred[:, i]))
        rows.append({'model': best, 'metric': f'{metric}_{t}', 'value': v})
        vals.append(v)
    mv = float(np.mean(vals))
    rows.append({'model': best, 'metric': f'{metric}_macro', 'value': mv})
    macro[metric] = mv

out = pd.DataFrame(rows)
out['value'] = out['value'].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(out, 'test_metrics.csv', caption='Test metrics')

mlca_runtime.record_metric(primary, macro[primary.lower()], split='test')
mlca_runtime.record_fact_candidate('test_score', macro[primary.lower()], source='test_metrics.csv')
print(f"best={best} primary={primary} test={macro[primary.lower()]:.4f}")
print(f"rows={len(out)} features={len(feat_cols)}")
