import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

targets = list(proto['targets'])
primary = proto['task']['primary_metric']
excluded = list(proto.get('excluded_columns', []))
group = proto.get('group_column', None)

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

raw_cols = [c for c in train.columns
            if c not in targets and c not in excluded and (group is None or c != group)]
keylike = []
for c in raw_cols:
    s = train[c]
    if s.is_unique and (s.dtype.kind in 'iu' or s.dtype == object or str(s.dtype).startswith('string')):
        keylike.append(c)
raw_cols = [c for c in raw_cols if c not in keylike]

with open(inp['05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_retain_features_YTKXSJV5/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)
with open(inp['07_model_train_baseline_1RE4HZWD/model.pkl'], 'rb') as f:
    m_base = pickle.load(f)
with open(inp['08_model_train_alternative_22ZP3DFW/alternative_model.pkl'], 'rb') as f:
    m_alt = pickle.load(f)

X_val = val[raw_cols].copy()
y_val = {t: val[t].to_numpy() for t in targets}

def first_input(obj):
    step = obj.steps[0][1] if isinstance(obj, Pipeline) else obj
    return set(map(str, getattr(step, 'feature_names_in_', [])))

raw_set = set(map(str, raw_cols))

def build(name, model):
    if first_input(model) == raw_set:
        return name, model
    for cand in [Pipeline([('pre', pre), ('sel', sel), ('est', model)]),
                 Pipeline([('pre', pre), ('est', model)])]:
        try:
            cand.predict(X_val.head(3))
            return name, cand
        except Exception:
            pass
    return name, Pipeline([('pre', pre), ('sel', sel), ('est', model)])

cands = [build('baseline', m_base), build('alternative', m_alt)]

def col(pred, i):
    if pred.ndim == 2 and pred.shape[1] == len(targets):
        return pred[:, i]
    return pred.ravel()

minimize = {'rmse', 'mae', 'mse', 'mean_squared_error', 'log_loss',
            'poisson_deviance', 'deviance'}
direction = 'minimize' if primary.lower() in minimize else 'maximize'

rows = []
scores = {}
for name, pipe in cands:
    pred = np.asarray(pipe.predict(X_val))
    per = []
    for i, t in enumerate(targets):
        p = col(pred, i); y = y_val[t]
        vals = {
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'mae': float(mean_absolute_error(y, p)),
            'r2': float(r2_score(y, p)),
        }
        for m, v in vals.items():
            rows.append({'model': name, 'metric': f'{m}_{t}', 'value': v})
        if primary.lower() in vals:
            per.append(vals[primary.lower()])
    for m in ('rmse', 'mae', 'r2'):
        per_vals = []
        for i, t in enumerate(targets):
            y = y_val[t]; p = col(pred, i)
            if m == 'rmse':
                per_vals.append(float(np.sqrt(mean_squared_error(y, p))))
            elif m == 'mae':
                per_vals.append(float(mean_absolute_error(y, p)))
            else:
                per_vals.append(float(r2_score(y, p)))
        rows.append({'model': name, 'metric': f'{m}_macro', 'value': float(np.mean(per_vals))})
    if per:
        scores[name] = float(np.mean(per))

dfm = pd.DataFrame(rows)
dfm['value'] = dfm['value'].astype(float).round(4)
mlca_runtime.save_table(dfm, 'validation_metrics.csv', caption='validation metrics per model')

best_name = (min if direction == 'minimize' else max)(scores, key=scores.get)
best_val = float(scores[best_name])
mlca_runtime.record_metric(primary, best_val, 'val')
mlca_runtime.record_fact_candidate('best_validation_score', best_val, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')

print(f"val_rows={len(val)} features={len(raw_cols)} candidates={len(cands)}")
print(f"primary={primary} direction={direction} best={best_name} value={best_val:.4f}")
