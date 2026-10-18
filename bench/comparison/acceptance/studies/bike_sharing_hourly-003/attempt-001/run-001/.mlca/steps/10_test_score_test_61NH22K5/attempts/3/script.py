import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
try:
    task = protocol['task']; primary = task['primary_metric']
    targets = list(protocol['targets']); excluded = list(protocol['excluded_columns'])
except KeyError as e:
    raise KeyError(f"Missing {e}; top-level keys: {sorted(protocol.keys())}") from e
group = protocol.get('group_column')
time_cols = set(task.get('time_columns', []))

train_df = pd.read_parquet(inp['train'])
excl = set(targets) | set(excluded) | ({group} if group else set())

def is_keylike(s, name):
    if name in time_cols or len(s) <= 1: return False
    if s.nunique(dropna=False) != len(s): return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)

feature_cols = [c for c in train_df.columns if c not in excl and not is_keylike(train_df[c], c)]

def higher_better(metric, task):
    for k in ('greater_is_better','higher_is_better','maximize'):
        if isinstance(task.get(k), bool): return task[k]
    d = str(task.get('direction','')).lower()
    if d in ('max','maximize','higher','higher_is_better','greater'): return True
    if d in ('min','minimize','lower','lower_is_better','less'): return False
    return metric.lower() in ('r2','accuracy','f1','roc_auc','auc','precision','recall')

val = pd.read_csv(inp['09_validate_score_validation_T0YRBVEV/validation_metrics.csv'])
low = {c.lower(): c for c in val.columns}
mc = next((low[k] for k in ['model','model_name','candidate','candidate_name','name','estimator'] if k in low), None)
if mc is None: raise KeyError(f"No model col: {list(val.columns)}")
better = higher_better(primary, task)
metc = next((low[k] for k in ['metric','metric_name','measure'] if k in low), None)
valc = next((low[k] for k in ['value','score','metric_value'] if k in low), None)
p = primary.lower()
if metc and valc:
    sub = val[val[metc].astype(str).str.lower() == p]
    if sub.empty: raise ValueError(f"{primary} not in validation metrics")
    if sub[mc].duplicated().any(): sub = sub.groupby(mc, as_index=False)[valc].mean()
    idx = sub[valc].idxmax() if better else sub[valc].idxmin()
    sel_name = str(sub.loc[idx, mc])
else:
    c = low.get(p)
    if c is None: raise ValueError(f"{primary} not in cols")
    idx = val[c].idxmax() if better else val[c].idxmin()
    sel_name = str(val.loc[idx, mc])

n = sel_name.lower()
model_path = inp['07_model_train_baseline_V4Y4XVKZ/model.pkl'] if 'baseline' in n else inp['08_model_train_alternative_DHX507BF/alternative_model.pkl']
with open(model_path,'rb') as f: pipe = pickle.load(f)

expected = getattr(pipe, 'feature_names_in_', None)
if expected is not None:
    expected = list(expected)
    missing = [c for c in expected if c not in train_df.columns]
    bad = [c for c in expected if c in excl]
    if missing or bad: raise ValueError(f"Schema mismatch missing={missing} excluded={bad}")
    feature_cols = expected

test_df = pd.read_parquet(inp['test'])
X_test = test_df[feature_cols].copy()
y_pred = np.asarray(pipe.predict(X_test))
y_true = test_df[targets].to_numpy()
if len(targets) == 1: y_true = y_true.reshape(-1,1)
if y_pred.ndim == 1: y_pred = y_pred.reshape(-1,1)
if y_pred.shape[1] != len(targets):
    raise ValueError(f"Pred shape {y_pred.shape} vs targets {targets}")

per = {}
for i,t in enumerate(targets):
    yt, yp = y_true[:,i], y_pred[:,i]
    per[t] = {'rmse': float(np.sqrt(mean_squared_error(yt,yp))),
              'mae': float(mean_absolute_error(yt,yp)),
              'r2': float(r2_score(yt,yp))}
metrics = ['rmse','mae','r2']
macro = {m: float(np.mean([per[t][m] for t in targets])) for m in metrics}

rows = []
for t in targets:
    for m in metrics:
        rows.append({'model': sel_name, 'metric': f'{m}_{t}', 'value': round(per[t][m],6)})
for m in metrics:
    rows.append({'model': sel_name, 'metric': f'{m}_macro', 'value': round(macro[m],6)})
out = pd.DataFrame(rows, columns=['model','metric','value'])
mlca_runtime.save_table(out, 'test_metrics.csv', caption='Test metrics per target and macro average')

alias = {'root_mean_squared_error':'rmse','mean_absolute_error':'mae','r2_score':'r2'}
pkey = alias.get(p, p)
if pkey not in macro: raise ValueError(f"Unsupported primary metric: {primary}")
pv = macro[pkey]
mlca_runtime.record_metric(primary, pv, split='test')
mlca_runtime.record_fact_candidate('test_score', pv, source='test_metrics.csv')
print(f"selected={sel_name} rows={len(test_df)} feats={len(feature_cols)}")
print(f"{primary}={pv:.6g}")
