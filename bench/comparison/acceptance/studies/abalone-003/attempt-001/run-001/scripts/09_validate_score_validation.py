import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
if 'targets' not in proto or 'task' not in proto:
    raise KeyError(f"Protocol missing targets/task; available keys: {list(proto)}")

targets = proto['targets']
primary = proto['task']['primary_metric']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')
val = pd.read_parquet(inp['val'])
feat_cols = [c for c in val.columns if c not in targets and c not in excluded and (group is None or c != group)]

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inp['05_preprocess_fit_preprocessing_QJRR3HF7/preprocessor.pkl'])
sel = load_pkl(inp['06_features_retain_features_8C5RPJ11/selector_all.pkl'])

def pre_has_selector(p):
    if isinstance(p, Pipeline):
        for name, obj in p.steps:
            if 'selector' in name.lower() or 'selector' in obj.__class__.__name__.lower():
                return True
    return False

def build_pipe(model):
    if isinstance(model, Pipeline):
        return model
    steps = []
    if pre is not None:
        steps.append(('preprocessor', pre))
    if sel is not None and not pre_has_selector(pre):
        steps.append(('selector', sel))
    steps.append(('model', model))
    return Pipeline(steps)

cands = {}
seen = set()
for label, key in [
    ('baseline', '07_model_train_baseline_3H582954/model.pkl'),
    ('alternative', '08_model_train_alternative_J64AD4FF/alternative_model.pkl'),
]:
    p = inp[key]
    rp = p.resolve()
    if rp not in seen:
        seen.add(rp)
        cands[label] = build_pipe(load_pkl(p))

def input_cols(pipe):
    if hasattr(pipe, 'feature_names_in_'):
        return list(pipe.feature_names_in_)
    if isinstance(pipe, Pipeline):
        for _, step in pipe.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return feat_cols

def eval_rows(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    rows = []
    for i, t in enumerate(targets):
        a, b = yt[:, i], yp[:, i]
        rows.append((t, 'rmse', float(np.sqrt(mean_squared_error(a, b)))))
        rows.append((t, 'mae', float(mean_absolute_error(a, b))))
        rows.append((t, 'r2', float(r2_score(a, b))))
    for m in ['rmse', 'mae', 'r2']:
        vals = [v for _, mm, v in rows if mm == m]
        rows.append(('macro', m, float(np.mean(vals))))
    return rows

all_rows = []
primary_vals = {}
for label, pipe in cands.items():
    cols = input_cols(pipe)
    cols = cols if all(c in val.columns for c in cols) else feat_cols
    pred = pipe.predict(val[cols].copy())
    rows = eval_rows(val[targets], pred)
    for t, m, v in rows:
        metric_name = f'{m}_{t}'
        all_rows.append({'model': label, 'metric': metric_name, 'value': v})
    primary_vals[label] = next(v for t, m, v in rows if t == 'macro' and m == primary)

lower = any(k in primary.lower() for k in ['rmse', 'mae', 'mse', 'log_loss', 'error'])
best_label = min(primary_vals, key=primary_vals.get) if lower else max(primary_vals, key=primary_vals.get)
best_val = primary_vals[best_label]

out = pd.DataFrame(all_rows)
out['value'] = out['value'].map(lambda v: float(f'{v:.4g}'))
mlca_runtime.save_table(out, 'validation_metrics.csv', caption='Validation metrics by candidate and target')
mlca_runtime.record_metric(primary, best_val, split='val')
mlca_runtime.record_fact_candidate('best_validation_score', best_val, source='validation_metrics.csv')

print(f'validation rows: {len(val)}')
print(f'candidates: {list(cands)}')
print(f'best {primary}: {best_label} = {best_val:.4g}')
