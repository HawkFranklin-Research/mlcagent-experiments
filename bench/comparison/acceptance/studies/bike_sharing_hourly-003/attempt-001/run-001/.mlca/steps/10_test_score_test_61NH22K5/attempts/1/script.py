import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
try:
    task = protocol['task']; primary = task['primary_metric']
    targets = protocol['targets']; excluded = protocol['excluded_columns']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {sorted(protocol.keys())}") from e
group = protocol.get('group_column')
time_cols = set(task.get('time_columns', []))

train_df = pd.read_parquet(inp['train'])
excl = set(targets) | set(excluded)
if group:
    excl.add(group)

def is_keylike(s, name):
    if name in time_cols or s.nunique(dropna=False) != len(s) or len(s) <= 1:
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)

feature_cols = [c for c in train_df.columns if c not in excl and not is_keylike(train_df[c], c)]

def higher_better(metric, task):
    for k in ('greater_is_better', 'higher_is_better', 'maximize'):
        if isinstance(task.get(k), bool):
            return task[k]
    d = str(task.get('direction', '')).lower()
    if d in ('max', 'maximize', 'higher', 'higher_is_better', 'greater'):
        return True
    if d in ('min', 'minimize', 'lower', 'lower_is_better', 'less'):
        return False
    return metric.lower() in ('r2', 'r2_score', 'accuracy', 'f1', 'f1_score', 'roc_auc', 'auc', 'precision', 'recall')

def pick_candidate(path, primary, task):
    df = pd.read_csv(path); low = {c.lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            if n in low:
                return low[n]
    model_col = find('model', 'model_name', 'candidate', 'candidate_name', 'name', 'estimator')
    if model_col is None:
        raise KeyError(f"No model column; columns={list(df.columns)}")
    better = higher_better(primary, task)
    metric_col = find('metric', 'metric_name', 'measure')
    value_col = find('value', 'score', 'metric_value')
    p = primary.lower()
    if metric_col and value_col:
        sub = df[df[metric_col].astype(str).str.lower() == p]
        if sub.empty:
            raise ValueError(f"Primary metric {primary} not in validation metrics")
        if sub[model_col].duplicated().any():
            sub = sub.groupby(model_col, as_index=False)[value_col].mean()
        idx = sub[value_col].idxmax() if better else sub[value_col].idxmin()
        return str(sub.loc[idx, model_col])
    c = low.get(p)
    if c is None:
        for k, v in low.items():
            if k.replace(' ', '_') == p.replace(' ', '_'):
                c = v; break
    if c is None:
        raise ValueError(f"Primary metric {primary} not in columns {list(df.columns)}")
    idx = df[c].idxmax() if better else df[c].idxmin()
    return str(df.loc[idx, model_col])

sel_name = pick_candidate(inp['09_validate_score_validation_T0YRBVEV/validation_metrics.csv'], primary, task)
n = sel_name.lower()
if 'baseline' in n:
    model_path = inp['07_model_train_baseline_V4Y4XVKZ/model.pkl']
else:
    model_path = inp['08_model_train_alternative_DHX507BF/alternative_model.pkl']

def load_obj(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

cand = load_obj(model_path)
def has_step(obj, names):
    return isinstance(obj, Pipeline) and any(k in obj.named_steps for k in names)

if has_step(cand, ['preprocessor', 'preprocess', 'preprocessing']) and has_step(cand, ['selector', 'feature_selector', 'select']):
    pipe = cand
else:
    pre = load_obj(inp['05_preprocess_fit_preprocessing_CK5QBBYX/preprocessor.pkl'])
    sel = load_obj(inp['06_features_retain_features_5J0R0BF8/selector_all.pkl'])
    if isinstance(cand, Pipeline):
        steps = list(cand.steps); names = [s[0] for s in steps]
        if not any(k in names for k in ['preprocessor', 'preprocess', 'preprocessing']):
            steps = [('preprocessor', pre)] + steps
        if not any(k in names for k in ['selector', 'feature_selector', 'select']):
            steps = steps[:-1] + [('selector', sel)] + steps[-1:]
        pipe = Pipeline(steps)
    else:
        pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', cand)])

expected = getattr(pipe, 'feature_names_in_', None)
if expected is not None:
    expected = list(expected)
    missing = [c for c in expected if c not in train_df.columns]
    bad = [c for c in expected if c in excl]
    if missing or bad:
        raise ValueError(f"Pipeline schema mismatch: missing={missing}, excluded={bad}")
    feature_cols = expected

test_df = pd.read_parquet(inp['test'])
X_test = test_df[feature_cols].copy()
y_pred = np.asarray(pipe.predict(X_test))
y_true = test_df[targets].to_numpy()
if len(targets) == 1:
    y_true = y_true.reshape(-1, 1)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)
if y_pred.shape[1] != len(targets):
    raise ValueError(f"Prediction shape {y_pred.shape} does not match targets {targets}")

metrics = ['rmse', 'mae', 'r2']
per = {}
for i, t in enumerate(targets):
    yt, yp = y_true[:, i], y_pred[:, i]
    per[t] = {
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mae': float(mean_absolute_error(yt, yp)),
        'r2': float(r2_score(yt, yp)),
    }
macro = {m: float(np.mean([per[t][m] for t in targets])) for m in metrics}
rows = []
for t in targets:
    for m in metrics:
        rows.append({'model': sel_name, 'metric': f'{m}_{t}', 'value': float(f'{per[t][m]:.4g}')})
for m in metrics:
    rows.append({'model': sel_name, 'metric': f'{m}_macro', 'value': float(f'{macro[m]:.4g}')})
out = pd.DataFrame(rows, columns=['model', 'metric', 'value'])
mlca_runtime.save_table(out, 'test_metrics.csv', caption='Test metrics per target and macro average')

alias = {'root_mean_squared_error': 'rmse', 'mean_absolute_error': 'mae', 'r2_score': 'r2'}
pkey = alias.get(primary.lower(), primary.lower())
if pkey not in macro:
    raise ValueError(f"Unsupported primary metric: {primary}")
primary_value = macro[pkey]
mlca_runtime.record_metric(primary, primary_value, split='test')
mlca_runtime.record_fact_candidate('test_score', primary_value, source='test_metrics.csv')
print(f"selected={sel_name}")
print(f"test_rows={len(test_df)} features={len(feature_cols)}")
print(f"{primary}={primary_value:.6g}")
