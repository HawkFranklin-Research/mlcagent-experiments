import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target_col = protocol['targets'][0]
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task['type']
positive_label = protocol.get('positive_label') or task.get('positive_label')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])
feature_cols = [c for c in train.columns
                if c not in protocol['targets'] and c not in excluded and c != group_col]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def cols_for(pipe):
    fni = getattr(pipe, 'feature_names_in_', None)
    return list(fni) if fni is not None else feature_cols

def probe(pipe):
    if not isinstance(pipe, Pipeline) or not hasattr(pipe, 'predict_proba'):
        return None
    cols = cols_for(pipe)
    try:
        pipe.predict_proba(val[cols].head(2))
        return cols
    except Exception:
        return None

cands = {}
for name, key in [('logistic', '07_model_train_logistic_BMNCC8GS/model.pkl'),
                  ('alternative', '08_model_train_alternative_KFV9SB0C/alternative_model.pkl'),
                  ('latest', 'latest_model')]:
    if key not in inputs:
        continue
    try:
        obj = load(inputs[key])
    except Exception:
        continue
    cols = probe(obj)
    if cols is not None:
        cands[name] = (obj, cols)

def classes_of(pipe):
    if hasattr(pipe, 'classes_'):
        return pipe.classes_
    return list(pipe.named_steps.values())[-1].classes_

def val_score(pipe, cols):
    proba = pipe.predict_proba(val[cols])
    cls = classes_of(pipe)
    pl = positive_label if positive_label else cls[1]
    if task_type == 'binary':
        y_bin = (val[target_col] == pl).astype(int)
        return roc_auc_score(y_bin, proba[:, int(np.where(cls == pl)[0][0])])
    return roc_auc_score(val[target_col], proba, multi_class='ovr',
                         average='macro', labels=cls)

best_name, best_score, best_pipe, best_cols = None, -np.inf, None, None
for name, (pipe, cols) in cands.items():
    try:
        s = val_score(pipe, cols)
        if s > best_score:
            best_score, best_name, best_pipe, best_cols = s, name, pipe, cols
    except Exception:
        pass

if best_pipe is None:
    return_keys = sorted(inputs.keys())
    raise RuntimeError(f"No complete pipeline among inputs; available: {return_keys}")

cls = classes_of(best_pipe)
pl = positive_label if positive_label else cls[1]
pos = int(np.where(cls == pl)[0][0])

threshold = None
try:
    vm = pd.read_csv(inputs['09_validate_score_validation_0H12DC4P/validation_metrics.csv'])
    if 'threshold' in vm.columns:
        threshold = float(vm['threshold'].iloc[0])
except Exception:
    pass
if threshold is None and task_type == 'binary':
    yv_prob = best_pipe.predict_proba(val[best_cols])[:, pos]
    yv_bin = (val[target_col] == pl).astype(int)
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        ba = balanced_accuracy_score(yv_bin, (yv_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    threshold = best_t

proba_test = best_pipe.predict_proba(test[best_cols])

if task_type == 'binary':
    y_vec = (test[target_col].values == pl).astype(int)
    scores = proba_test[:, pos]
    def compute(yt, ys):
        yp = (ys >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return {'roc_auc': roc_auc_score(yt, ys),
                'pr_auc': average_precision_score(yt, ys),
                'balanced_accuracy': balanced_accuracy_score(yt, yp),
                'brier': brier_score_loss(yt, ys),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else float('nan'),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else float('nan')}
    point = compute(y_vec, scores)
else:
    y_vec = test[target_col].values
    scores = proba_test
    def compute(yt, ys):
        yp = cls[np.argmax(ys, axis=1)]
        return {'roc_auc_ovr': roc_auc_score(yt, ys, multi_class='ovr', average='macro', labels=cls),
                'macro_f1': f1_score(yt, yp, average='macro')}
    point = compute(y_vec, scores)

rng = np.random.RandomState(42)
n = len(y_vec)
boots = {k: [] for k in point}
for _ in range(1000):
    idx = rng.randint(0, n, n)
    yt, ys = y_vec[idx], scores[idx]
    if task_type == 'binary':
        if len(np.unique(yt)) < 2:
            continue
    else:
        if len(np.unique(yt)) < len(cls):
            continue
    try:
        m = compute(yt, ys)
        for k, v in m.items():
            if np.isfinite(v):
                boots[k].append(v)
    except Exception:
        pass

rows = []
for k, v in point.items():
    bl = boots[k]
    lo = float(np.percentile(bl, 2.5)) if len(bl) > 1 else float('nan')
    hi = float(np.percentile(bl, 97.5)) if len(bl) > 1 else float('nan')
    rows.append({'metric': k, 'value': v, 'ci_low': lo, 'ci_high': hi})

df = pd.DataFrame(rows)
for c in ['value', 'ci_low', 'ci_high']:
    df[c] = df[c].apply(lambda x: float(f'{x:.4g}') if pd.notnull(x) else x)
mlca_runtime.save_table(df, 'test_metrics.csv',
                        caption='Test metrics with 95% bootstrap CI')

row = df[df['metric'] == primary_metric].iloc[0]
mlca_runtime.record_metric(primary_metric, float(row['value']), split='test',
                           ci=[float(row['ci_low']), float(row['ci_high'])])
if primary_metric == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(row['value']),
                                       source='test_metrics.csv')

print(f"Selected: {best_name}, val {primary_metric}: {best_score:.4f}")
print(f"Test {primary_metric}: {row['value']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
