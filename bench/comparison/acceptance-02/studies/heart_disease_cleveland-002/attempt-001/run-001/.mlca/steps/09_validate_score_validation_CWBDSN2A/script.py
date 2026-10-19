import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
try:
    targets = list(proto['targets'])
    primary = str(proto['task']['primary_metric'])
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto.keys())}") from None
excluded = list(proto.get('excluded_columns', []))
group = proto.get('group_column')
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

def _load(k):
    with open(inp[k], 'rb') as f:
        return pickle.load(f)

pre = _load('05_preprocess_fit_preprocessing_5Z83A35F/preprocessor.pkl')
sel = _load('06_features_retain_features_Z9TAK0X7/selector_all.pkl')
models = {
    'baseline': _load('07_model_train_baseline_AHTSSAGQ/model.pkl'),
    'alternative': _load('08_model_train_alternative_AC8TT8JH/alternative_model.pkl'),
}

def _names(o):
    n = getattr(o, 'feature_names_in_', None)
    if n is None and isinstance(o, Pipeline):
        for s in o.named_steps.values():
            n = getattr(s, 'feature_names_in_', None)
            if n is not None:
                break
    return list(n) if n is not None else None

def _front():
    if isinstance(sel, Pipeline) and len(sel.named_steps):
        first_name, first = next(iter(sel.named_steps.items()))
        if hasattr(first, 'transformers') or 'pre' in first_name.lower():
            return sel
    if pre is not None and sel is not None:
        return Pipeline([('pre', pre), ('sel', sel)])
    return pre if pre is not None else sel

def _predictor(m):
    if isinstance(m, Pipeline) and len(m.named_steps) > 1:
        return m
    f = _front()
    return m if f is None else Pipeline([('frontend', f), ('model', m)])

def _X(pred):
    n = _names(pred)
    if n and all(c in val.columns for c in n) and not any(c in targets or c in excluded for c in n):
        return val[list(n)]
    return val[feature_cols]

cats = {}
for t in targets:
    vals = pd.unique(train[t].dropna())
    try:
        vals = np.sort(vals)
    except Exception:
        vals = list(vals)
    cats[t] = list(vals)

def _score(pred, X):
    p = pred.predict(X)
    p = np.column_stack(p) if isinstance(p, list) else np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    true_coded, pred_coded, per_f1, per_hl = [], [], [], []
    for j, t in enumerate(targets):
        yt = val[t].to_numpy()
        yp = p[:, j]
        ct = pd.Categorical(yt, categories=cats[t]).codes
        cp = pd.Categorical(yp, categories=cats[t]).codes
        true_coded.append(ct)
        pred_coded.append(cp)
        per_f1.append(f1_score(ct, cp, average='macro'))
        per_hl.append(hamming_loss(ct, cp))
    macro = float(np.mean(per_f1))
    hl = float(np.mean(per_hl))
    subset = float(np.mean(np.all(np.column_stack(pred_coded) == np.column_stack(true_coded), axis=1)))
    return macro, hl, subset, per_f1, per_hl

rows, scores = [], {}
for name, model in models.items():
    pred = _predictor(model)
    macro, hl, subset, per_f1, per_hl = _score(pred, _X(pred))
    scores[name] = macro
    vals = [(primary, macro), ('hamming_loss', hl), ('subset_accuracy', subset)]
    vals += [(f'f1_{t}', v) for t, v in zip(targets, per_f1)]
    vals += [(f'hamming_{t}', v) for t, v in zip(targets, per_hl)]
    for metric, v in vals:
        rows.append({'model': name, 'metric': metric, 'value': float(f'{v:.4g}')})

report = pd.DataFrame(rows)
mlca_runtime.save_table(report, 'validation_metrics.csv', caption='Validation candidate metrics')
lower_better = 'loss' in primary.lower() or primary.lower() in ('rmse', 'mae', 'mse')
best_name = min(scores, key=scores.get) if lower_better else max(scores, key=scores.get)
best_score = scores[best_name]
mlca_runtime.record_metric(primary, float(best_score), 'val')
mlca_runtime.record_fact_candidate('best_validation_score', float(best_score), source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
print(train.shape, val.shape)
print('best', best_name, f'{best_score:.4g}')
