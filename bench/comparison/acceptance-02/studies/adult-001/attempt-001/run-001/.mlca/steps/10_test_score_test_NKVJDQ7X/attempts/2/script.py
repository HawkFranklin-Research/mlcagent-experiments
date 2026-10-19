import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
task = proto['task']
primary = task['primary_metric']
targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns') or [])
group = proto.get('group_column', None)
target = targets[0]
positive = proto.get('positive_label', None)

train = pd.read_parquet(inputs['train'])
val   = pd.read_parquet(inputs['val'])
test  = pd.read_parquet(inputs['test'])

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inputs['05_preprocess_fit_preprocessing_ABA216ED/preprocessor.pkl'])
sel = load_pkl(inputs['06_features_select_features_DJ7CCJDF/selector.pkl'])
ma  = load_pkl(inputs['07_model_train_logistic_X4STNM7K/model.pkl'])
mb  = load_pkl(inputs['08_model_train_alternative_KMW9HJFX/alternative_model.pkl'])

man09 = json.loads(inputs['09_validate_score_validation_8Q1BNG23/manifest.json'].read_text())
vm = pd.read_csv(inputs['09_validate_score_validation_8Q1BNG23/validation_metrics.csv'])

def to_pipe(obj):
    if isinstance(obj, Pipeline) and len(obj.named_steps) >= 2:
        return obj
    return Pipeline([('pre', pre), ('sel', sel), ('est', obj)])

pipe_a = to_pipe(ma)
pipe_b = to_pipe(mb)

def feat_cols(pipe):
    for src in [pipe] + (list(pipe.named_steps.values()) if isinstance(pipe, Pipeline) else []):
        if hasattr(src, 'feature_names_in_'):
            return list(src.feature_names_in_)
    return [c for c in test.columns if c not in targets and c not in excluded and c != group]

cols = feat_cols(pipe_a)
va_x = val[cols].copy()
te_x = test[cols].copy()
y_val = (val[target].astype(str) == str(positive)).astype(int).values if positive is not None else None

classes_a = list(pipe_a.classes_)
pos = positive if (positive is not None and positive in classes_a) else classes_a[1]
j = classes_a.index(pos)
y_val = (val[target] == pos).astype(int).values
y_test = (test[target] == pos).astype(int).values

def val_auc(pipe, c):
    p = pipe.predict_proba(va_x)[:, c]
    return roc_auc_score(y_val, p)

score_a = val_auc(pipe_a, j)
score_b = val_auc(pipe_b, j)

chosen_name = None
if {'model', 'metric'}.issubset(vm.columns):
    sub = vm[vm['metric'] == primary]
    if 'split' in vm.columns and (vm['split'] == 'val').any():
        sub = sub[sub['split'] == 'val']
    if len(sub):
        sub = sub.sort_values('value', ascending=False)
        chosen_name = str(sub.iloc[0]['model'])

if chosen_name and '07' in chosen_name or (chosen_name and 'logistic' in chosen_name.lower()):
    chosen, jj = pipe_a, j
elif chosen_name and ('08' in chosen_name or 'alt' in chosen_name.lower()):
    chosen, jj = pipe_b, list(pipe_b.classes_).index(pos)
else:
    if score_a >= score_b:
        chosen, jj = pipe_a, j
    else:
        chosen, jj = pipe_b, list(pipe_b.classes_).index(pos)

thr = None
for src in (man09, task, proto):
    if isinstance(src, dict):
        for k in ('threshold', 'decision_threshold', 'frozen_threshold'):
            if src.get(k) is not None:
                thr = float(src[k]); break
    if thr is not None:
        break
if thr is None:
    p_val = chosen.predict_proba(va_x)[:, jj]
    fpr, tpr, thrs = roc_curve(y_val, p_val)
    thr = float(thrs[np.argmax(tpr - fpr)])

p_test = chosen.predict_proba(te_x)[:, jj]
pred_test = (p_test >= thr).astype(int)

def compute_metrics(y, p, pred):
    m = {
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'brier_score': float(brier_score_loss(y, p)),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
    return m

point = compute_metrics(y_test, p_test, pred_test)
rng = np.random.default_rng(42)
n = len(y_test)
boot = {k: [] for k in point}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb = y_test[idx], p_test[idx]
    if len(np.unique(yb)) < 2:
        continue
    bm = compute_metrics(yb, pb, (pb >= thr).astype(int))
    for k, v in bm.items():
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for k, v in point.items():
    arr = np.array(boot[k]) if boot[k] else np.array([v])
    rows.append({'metric': k, 'value': round(float(v), 6),
                 'ci_low': round(float(np.percentile(arr, 2.5)), 6),
                 'ci_high': round(float(np.percentile(arr, 97.5)), 6)})
tm = pd.DataFrame(rows)
mlca_runtime.save_table(tm, 'test_metrics.csv',
    caption='Test metrics with 95% bootstrap CI (1000 resamples).')

prim_row = tm[tm['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prim_row['value']), 'test',
    ci=[float(prim_row['ci_low']), float(prim_row['ci_high'])])
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(prim_row['value']),
        source='test_metrics.csv')

print('test rows', len(test), 'pos_label', pos, 'thr', round(thr, 4))
print('primary', primary, round(float(prim_row['value']), 4),
      'ci', round(float(prim_row['ci_low']), 4), round(float(prim_row['ci_high']), 4))
