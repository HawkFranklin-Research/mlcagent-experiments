import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs['_protocol'].read_text())
task = proto['task']
primary = task['primary_metric']
targets = proto['targets']
excluded = list(proto.get('excluded_columns', []))
group = proto.get('group_column', None)
target = targets[0]
positive = proto.get('positive_label', None)

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl(inputs['05_preprocess_fit_preprocessing_ABA216ED/preprocessor.pkl'])
selector     = load_pkl(inputs['06_features_select_features_DJ7CCJDF/selector.pkl'])
model_a      = load_pkl(inputs['07_model_train_logistic_X4STNM7K/model.pkl'])
model_b      = load_pkl(inputs['08_model_train_alternative_KMW9HJFX/alternative_model.pkl'])

man09 = json.loads(inputs['09_validate_score_validation_8Q1BNG23/manifest.json'].read_text())

# Decide chosen candidate from validation manifest / metrics (validation only)
vm = pd.read_csv(inputs['09_validate_score_validation_8Q1BNG23/validation_metrics.csv'])

def pick_from_metrics(df, metric_name):
    d = df[df['metric'] == metric_name] if 'metric' in df.columns else df
    if 'split' in df.columns:
        d = d[d['split'] == 'val']
    higher = metric_name in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'f1', 'f1_macro', 'sensitivity')
    d = d.sort_values('value', ascending=not higher)
    return d.iloc[0] if len(d) else None

# Determine which fitted pipeline is the frozen model
cand_a, cand_b = model_a, model_b
try:
    row = pick_from_metrics(vm, primary)
    chosen_name = str(row.get('model', row.get('candidate', ''))) if row is not None else ''
except Exception:
    chosen_name = ''

def full_pipe(obj):
    return isinstance(obj, Pipeline)

def build(est):
    if full_pipe(est):
        return est
    # compose selector + est (check whether selector wraps preprocessing)
    if full_pipe(selector):
        steps = [('sel', selector), ('est', est)]
    else:
        steps = [('pre', preprocessor), ('sel', selector), ('est', est)]
    return Pipeline(steps)

pipe_a = build(model_a)
pipe_b = build(model_b)

# evaluate the two on val to (re)confirm winner without using test
def val_auc(pipe):
    if positive is not None:
        classes = list(pipe.classes_) if hasattr(pipe, 'classes_') else list(pipe.named_steps[list(pipe.named_steps)[-1]].classes_)
        pos = positive if positive in classes else classes[1]
        j = classes.index(pos)
        y = (val[target] == pos).astype(int).values
        p = pipe.predict_proba(val[feature_cols_check])[:, j] if False else pipe.predict_proba(val)[:, j]
    else:
        classes = list(pipe.named_steps[list(pipe.named_steps)[-1]].classes_)
        j = 1
        pos = classes[1]
        y = (val[target] == pos).astype(int).values
        p = pipe.predict_proba(val)[:, j]
    return roc_auc_score(y, p)

score_a = val_auc(pipe_a)
score_b = val_auc(pipe_b)
# prioritize explicit chosen_name if it names one; else use higher val score
if chosen_name:
    low = chosen_name.lower()
    if 'alt' in low or '08' in low:
        chosen = pipe_b
    elif 'logistic' in low or '07' in low:
        chosen = pipe_a
    else:
        chosen = pipe_a if score_a >= score_b else pipe_b
else:
    chosen = pipe_a if score_a >= score_b else pipe_b

# frozen validation threshold: look for it in validation manifest, else derive on val (Youden J)
thr = None
for src in (man09, proto.get('task', {}), proto):
    if isinstance(src, dict):
        for k in ('threshold', 'decision_threshold', 'frozen_threshold'):
            if k in src and src[k] is not None:
                thr = float(src[k]); break
    if thr is not None: break

classes_chosen = list(chosen.named_steps[list(chosen.named_steps)[-1]].classes_) if isinstance(chosen, Pipeline) else list(chosen.classes_)
if positive is not None and positive in classes_chosen:
    pos_label = positive
else:
    pos_label = classes_chosen[1]
pos_idx = classes_chosen.index(pos_label)

# on val, compute threshold if not frozen
if thr is None:
    y_val = (val[target] == pos_label).astype(int).values
    p_val = chosen.predict_proba(val)[:, pos_idx]
    from sklearn.metrics import roc_curve
    fpr, tpr, thrs = roc_curve(y_val, p_val)
    thr = float(thrs[np.argmax(tpr - fpr)])

y_test = (test[target] == pos_label).astype(int).values
p_test = chosen.predict_proba(test)[:, pos_idx]
pred_test = (p_test >= thr).astype(int)

def metrics_from(y, p, pred):
    m = {
        'roc_auc': roc_auc_score(y, p),
        'pr_auc': average_precision_score(y, p),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'brier_score': brier_score_loss(y, p),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    m['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    return m

point = metrics_from(y_test, p_test, pred_test)

# bootstrap
rng = np.random.default_rng(42)
n = len(y_test)
boot = {k: [] for k in point}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb = y_test[idx], p_test[idx]
    if len(np.unique(yb)) < 2:
        continue
    pb_bin = (pb >= thr).astype(int)
    bm = metrics_from(yb, pb, pb_bin)
    for k, v in bm.items():
        if not (isinstance(v, float) and np.isnan(v)):
            boot[k].append(v)

rows = []
for k, v in point.items():
    arr = np.array(boot[k])
    lo = float(np.percentile(arr, 2.5)) if len(arr) > 0 else float('nan')
    hi = float(np.percentile(arr, 97.5)) if len(arr) > 0 else float('nan')
    rows.append({'metric': k, 'value': float(v), 'ci_low': lo, 'ci_high': hi})

tm = pd.DataFrame(rows)
path = mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI (1000 resamples).')

# record primary metric with CI
prim_row = tm[tm['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prim_row['value']), 'test',
    ci=[float(prim_row['ci_low']), float(prim_row['ci_high'])])

if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(prim_row['value']),
        source='test_metrics.csv')

print('test rows', len(test), 'pos_label', pos_label, 'thr', round(thr, 4))
print('primary', primary, round(float(prim_row['value']), 4),
      'ci', round(float(prim_row['ci_low']), 4), round(float(prim_row['ci_high']), 4))
