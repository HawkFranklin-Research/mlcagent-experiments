import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

try:
    primary_metric = protocol['task']['primary_metric']
    targets = list(protocol['targets'])
    excluded = list(protocol['excluded_columns'])
    group_col = protocol.get('group_column')
except KeyError as e:
    raise RuntimeError(f"Missing key {e}; available: {list(protocol.keys())}")

with open(inp['latest_model'], 'rb') as f:
    model = pickle.load(f)
if not hasattr(model, 'predict_proba'):
    raise RuntimeError(f"Model lacks predict_proba; type={type(model)}")

test = pd.read_parquet(inp['test'])
fn = getattr(model, 'feature_names_in_', None)
if fn is not None:
    feat_in = list(fn)
else:
    base = [c for c in test.columns if c not in targets and c not in excluded and c != group_col]
    feat_in = []
    for c in base:
        s = test[c]
        is_str = s.dtype == object or pd.api.types.is_string_dtype(s)
        if (is_str or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=True) == len(s) and len(s) > 0:
            continue
        feat_in.append(c)

X_test = test[feat_in]
y_true = test[targets[0]]
classes = model.classes_
proba = model.predict_proba(X_test)

task = protocol['task']
pos_label = next((task[k] for k in ('positive_label', 'positive_class', 'positive')
                  if k in task and task[k] in list(classes)), classes[1])
pos_idx = list(classes).index(pos_label)
p_pos = proba[:, pos_idx]
y_bin = (y_true == pos_label).astype(int).to_numpy()

def find_thr(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ('threshold', 'decision_threshold', 'selected_threshold') \
               and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for it in obj:
            r = find_thr(it)
            if r is not None:
                return r
    return None

thr = 0.5
try:
    vm = json.loads(inp['09_validate_score_validation_4RY77ZSV/manifest.json'].read_text())
    tv = find_thr(vm)
    if tv is not None:
        thr = tv
except Exception:
    pass
if hasattr(model, 'steps'):
    final = model.steps[-1][1]
    for a in ('threshold_', 'threshold', 'decision_threshold_'):
        if hasattr(final, a):
            try:
                thr = float(getattr(final, a))
            except (TypeError, ValueError):
                pass
            break

y_pred_bin = (p_pos >= thr).astype(int)
y_pred_labels = np.where(p_pos >= thr, classes[pos_idx], classes[1 - pos_idx])

roc = roc_auc_score(y_bin, p_pos)
pr = average_precision_score(y_bin, p_pos)
brier = brier_score_loss(y_bin, p_pos)
bal = balanced_accuracy_score(y_bin, y_pred_bin)
tn, fp, fn_, tp = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1]).ravel()
sens = float(tp / (tp + fn_)) if (tp + fn_) > 0 else float('nan')
spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')

rng = np.random.RandomState(42)
n = len(y_bin)
bs = {k: [] for k in ('roc_auc', 'pr_auc', 'balanced_accuracy',
                      'brier_score', 'sensitivity', 'specificity')}
for _ in range(1000):
    idx = rng.randint(0, n, n)
    yb, pb = y_bin[idx], p_pos[idx]
    if len(np.unique(yb)) < 2:
        continue
    ypr = (pb >= thr).astype(int)
    try:
        bs['roc_auc'].append(roc_auc_score(yb, pb))
        bs['pr_auc'].append(average_precision_score(yb, pb))
        bs['brier_score'].append(brier_score_loss(yb, pb))
        bs['balanced_accuracy'].append(balanced_accuracy_score(yb, ypr))
        tn_, fp_, fn2, tp_ = confusion_matrix(yb, ypr, labels=[0, 1]).ravel()
        bs['sensitivity'].append(tp_ / (tp_ + fn2) if (tp_ + fn2) > 0 else np.nan)
        bs['specificity'].append(tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else np.nan)
    except Exception:
        continue

def ci(vals):
    a = []
    for v in vals:
        fv = float(v)
        if np.isfinite(fv):
            a.append(fv)
    if not a:
        return float('nan'), float('nan')
    arr = np.array(a)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

def r4(x):
    try:
        f = float(x)
        return float(f"{f:.4g}") if np.isfinite(f) else float('nan')
    except (TypeError, ValueError):
        return float('nan')

vals = {'roc_auc': roc, 'pr_auc': pr, 'balanced_accuracy': bal,
        'brier_score': brier, 'sensitivity': sens, 'specificity': spec}
rows = []
for k, v in vals.items():
    lo, hi = ci(bs[k])
    rows.append({'metric': k, 'value': r4(v), 'ci_low': r4(lo), 'ci_high': r4(hi)})
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv",
                        caption="Test metrics with 95% bootstrap CI (1000 resamples, seed 42)")

roc_lo, roc_hi = ci(bs['roc_auc'])
mlca_runtime.record_metric(primary_metric, float(roc), "test", ci=(roc_lo, roc_hi))
mlca_runtime.record_fact_candidate("test_roc_auc", float(roc), source="test_metrics.csv")

preds = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
preds[f'y_true__{targets[0]}'] = y_true.to_numpy()
preds[f'y_pred__{targets[0]}'] = y_pred_labels
for j, c in enumerate(classes):
    preds[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(preds, "test_predictions.parquet")

print(f"test {test.shape} thr={thr:.4f} pos={pos_label} metric={primary_metric}")
print(f"roc_auc={roc:.4f} [{roc_lo:.4f},{roc_hi:.4f}]")
