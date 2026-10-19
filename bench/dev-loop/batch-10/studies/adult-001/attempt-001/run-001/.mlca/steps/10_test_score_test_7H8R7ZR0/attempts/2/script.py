import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, recall_score, f1_score, roc_curve)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
target_col = protocol['targets'][0]
primary_metric = task['primary_metric']
task_type = task.get('type', 'binary')
positive_label = task.get('positive_label', protocol.get('positive_label', None))

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)
classes = list(model.classes_)

def get_feats(m):
    f = getattr(m, 'feature_names_in_', None)
    if f is None and hasattr(m, 'named_steps'):
        for s in m.named_steps.values():
            f = getattr(s, 'feature_names_in_', None)
            if f is not None:
                break
    if f is None:
        raise ValueError('model has no feature_names_in_')
    return list(f)

feats = get_feats(model)

def proba(df):
    return model.predict_proba(df[feats])

def prep(df):
    df = df[df[target_col].notna()].reset_index(drop=True)
    p = proba(df)
    mask = np.all(np.isfinite(p), axis=1)
    return df[mask].reset_index(drop=True), p[mask]

def align_y(y):
    if len(y) == 0 or not classes:
        return y
    c0 = classes[0]
    if isinstance(c0, str) and not isinstance(y[0], str):
        return np.array([str(v) for v in y])
    if isinstance(c0, (int, float, np.integer, np.floating)) and isinstance(y[0], str):
        try:
            return np.array([float(v) for v in y])
        except Exception:
            return y
    return y

pos = positive_label if (positive_label is not None and positive_label in classes) \
      else (classes[1] if len(classes) > 1 else classes[0])
pos_idx = classes.index(pos) if pos in classes else 0

def metrics(y_true, p, thr):
    if task_type == 'binary':
        pv = p[:, pos_idx]
        yb = (y_true == pos).astype(int)
        if len(np.unique(yb)) < 2:
            raise ValueError(f'positive label {pos!r} absent from y_true')
        m = {
            'roc_auc': float(roc_auc_score(yb, pv)),
            'pr_auc': float(average_precision_score(yb, pv)),
            'brier_score': float(brier_score_loss(yb, pv)),
        }
        yp = (pv >= thr).astype(int)
        m['balanced_accuracy'] = float(balanced_accuracy_score(yb, yp))
        m['sensitivity'] = float(recall_score(yb, yp, pos_label=1, zero_division=0))
        m['specificity'] = float(recall_score(yb, yp, pos_label=0, zero_division=0))
        return m
    aucs = []
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) >= 2:
            aucs.append(float(roc_auc_score(yb, p[:, j])))
    yp = np.array(classes)[np.argmax(p, axis=1)]
    return {'roc_auc_ovr': float(np.mean(aucs)) if aucs else float('nan'),
            'f1_macro': float(f1_score(y_true, yp, average='macro', zero_division=0))}

val, p_val = prep(pd.read_parquet(inputs['val']))
test, p_test = prep(pd.read_parquet(inputs['test']))
y_val = align_y(val[target_col].values)
y_test = align_y(test[target_col].values)

thr = 0.5
if task_type == 'binary':
    yb_val = (y_val == pos).astype(int)
    if len(np.unique(yb_val)) == 2:
        fpr, tpr, tvals = roc_curve(yb_val, p_val[:, pos_idx])
        t = tvals[np.argmax(tpr - fpr)]
        if np.isfinite(t):
            thr = float(t)

point = metrics(y_test, p_test, thr)
rng = np.random.default_rng(42)
n = len(y_test)
boot = {k: [] for k in point}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    try:
        m = metrics(y_test[idx], p_test[idx], thr)
    except Exception:
        continue
    for k, v in m.items():
        if np.isfinite(v):
            boot[k].append(v)

rows = []
for k, v in point.items():
    arr = np.array(boot.get(k, []))
    lo, hi = (np.percentile(arr, [2.5, 97.5])
              if (np.isfinite(v) and len(arr) >= 2) else (np.nan, np.nan))
    rows.append({'metric': k,
                 'value': float(v) if np.isfinite(v) else np.nan,
                 'ci_low': float(lo), 'ci_high': float(hi)})
df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df, 'test_metrics.csv', 'Test metrics with 95% bootstrap CI')

hit = df[df['metric'] == primary_metric]
if len(hit) == 0:
    raise ValueError(f'primary_metric {primary_metric!r} missing; have {list(df["metric"])}')
rv = float(hit.iloc[0]['value'])
if not np.isfinite(rv):
    raise ValueError(f'non-finite {primary_metric} on test after repair')
lo, hi = float(hit.iloc[0]['ci_low']), float(hit.iloc[0]['ci_high'])
ci = (lo, hi) if (np.isfinite(lo) and np.isfinite(hi)) else None
mlca_runtime.record_metric(primary_metric, rv, 'test', ci=ci)
if primary_metric == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', rv, source='test_metrics.csv')
print(f'test {primary_metric}={rv:.4f} n={n} thr={thr:.4f}')
