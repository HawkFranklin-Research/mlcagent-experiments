import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']; metric_name = task['primary_metric']; targets = protocol['targets']
excl_raw = protocol['excluded_columns']
if isinstance(excl_raw, dict):
    excluded = set()
    for k in ('columns','list','values','entries'):
        if isinstance(excl_raw.get(k), list): excluded |= set(excl_raw[k])
else:
    excluded = set(excl_raw)

vm = pd.read_csv(inp['09_validate_score_validation_ZMYHZAJZ/validation_metrics.csv'])
models_map = {'07_model_train_logistic_VFJ1YWNX': inp['07_model_train_logistic_VFJ1YWNX/model.pkl'],
              '08_model_train_alternative_7GE7PGA8': inp['08_model_train_alternative_7GE7PGA8/alternative_model.pkl']}
best_name = '07_model_train_logistic_VFJ1YWNX'
try:
    if 'metric' in vm.columns and 'model' in vm.columns:
        sel = vm[vm['metric']==metric_name]
        if len(sel): best_name = sel.sort_values('value', ascending=False).iloc[0]['model']
except Exception:
    pass
if best_name not in models_map: best_name = '07_model_train_logistic_VFJ1YWNX'
with open(models_map[best_name], 'rb') as f: model = pickle.load(f)

threshold = 0.5
try:
    vman = json.loads(inp['09_validate_score_validation_ZMYHZAJZ/manifest.json'].read_text())
    for k in ('threshold','decision_threshold','positive_threshold','best_threshold'):
        if k in vman: threshold = float(vman[k]); break
except Exception:
    pass

test = pd.read_parquet(inp['test'])
X_test = test[list(model.feature_names_in_)]
classes = list(model.classes_)
proba = np.asarray(model.predict_proba(X_test))
y_true_raw = test[targets[0]].values

pos = classes[1]; j_pos = classes.index(pos)
p_pos = proba[:, j_pos]
pr = (p_pos >= threshold).astype(int)

def _roc(idx):
    y = (y_true_raw[idx]==pos).astype(int)
    if y.min()==y.max(): return np.nan
    return roc_auc_score(y, p_pos[idx])
def _pr(idx):
    y = (y_true_raw[idx]==pos).astype(int)
    if y.min()==y.max(): return np.nan
    return average_precision_score(y, p_pos[idx])
def _ba(idx): return balanced_accuracy_score((y_true_raw[idx]==pos).astype(int),(p_pos[idx]>=threshold).astype(int))
def _brier(idx): return brier_score_loss((y_true_raw[idx]==pos).astype(int), p_pos[idx])
def _sens(idx):
    y=(y_true_raw[idx]==pos).astype(int); p=(p_pos[idx]>=threshold).astype(int)
    tp=((p==1)&(y==1)).sum(); fn=((p==0)&(y==1)).sum()
    return tp/(tp+fn) if (tp+fn)>0 else np.nan
def _spec(idx):
    y=(y_true_raw[idx]==pos).astype(int); p=(p_pos[idx]>=threshold).astype(int)
    tn=((p==0)&(y==0)).sum(); fp=((p==1)&(y==0)).sum()
    return tn/(tn+fp) if (tn+fp)>0 else np.nan

names = ['roc_auc','pr_auc','balanced_accuracy','brier_score','sensitivity','specificity']
fns = [_roc,_pr,_ba,_brier,_sens,_spec]
rng = np.random.default_rng(42); n = len(y_true_raw)
boot = {nm: [] for nm in names}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    for nm, fn in zip(names, fns):
        try:
            v = fn(idx)
            if np.isfinite(v): boot[nm].append(v)
        except Exception:
            pass

rows = []
for nm, fn in zip(names, fns):
    val = fn(np.arange(n))
    lo, hi = (np.percentile(boot[nm],[2.5,97.5]) if boot[nm] else (np.nan,np.nan))
    rows.append({'metric':nm,'value':round(float(val),6),'ci_low':round(float(lo),6),'ci_high':round(float(hi),6)})
    mlca_runtime.record_metric(nm, float(val), 'test', ci=(float(lo),float(hi)))
tm = pd.DataFrame(rows)
mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI (1000 resamples, seed 42)')

pred_label = np.where(pr==1, pos, classes[0])
pred_df = pd.DataFrame({'row_id': test.index, 'partition':'test', 'y_true': y_true_raw, 'y_pred': pred_label})
for j,c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

if metric_name == 'roc_auc':
    r = tm[tm['metric']=='roc_auc'].iloc[0]
    mlca_runtime.record_fact_candidate('test_roc_auc', float(r['value']), source='test_metrics.csv')

print(f"test {len(test)} rows, {X_test.shape[1]} features")
print(f"selected {best_name}, threshold={threshold}")
print(f"test {metric_name}={float(tm[tm['metric']==metric_name].iloc[0]['value']):.4f}")
