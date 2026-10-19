import json, pickle, numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
try:
    task = proto['task']; primary = task['primary_metric']
    targets = proto['targets']; excluded = proto['excluded_columns']
except KeyError as e:
    raise KeyError(f"Missing {e}; available top-level keys: {list(proto)}") from e
group = proto.get('group_column')
train = pd.read_parquet(inp['train']); test = pd.read_parquet(inp['test'])

def key_like(s):
    if not (s.notna().all() and s.nunique(dropna=True) == len(s)):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_integer_dtype(s) and any(k in str(s.name).lower() for k in ('id','key','code','seq','index'))

drop_cols = set(targets + excluded + ([group] if group else []))
raw_cols = [c for c in train.columns if c not in drop_cols and not key_like(train[c])]

def load_json(p): return json.loads(p.read_text())
val_metrics = pd.read_csv(inp['09_validate_score_validation_GQ48BZ4K/validation_metrics.csv'])
val_manifest = load_json(inp['09_validate_score_validation_GQ48BZ4K/manifest.json'])

def find(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in keys and isinstance(v, (str, int, float)): return v
            r = find(v, keys)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, keys)
            if r is not None: return r
    return None

sel_name = find(val_manifest, {'selected_model','selected_candidate','best_model','model_name','selected_estimator'})
threshold = find(val_manifest, {'threshold','decision_threshold','best_threshold','youden_threshold'})
if threshold is None:
    for c in val_metrics.columns:
        if 'threshold' in c.lower():
            v = pd.to_numeric(val_metrics[c], errors='coerce').dropna()
            if len(v): threshold = float(v.iloc[0]); break
if threshold is None and {'metric','value'}.issubset(val_metrics.columns):
    m = val_metrics[val_metrics['metric'].astype(str).str.lower() == 'threshold']
    if len(m): threshold = float(m['value'].iloc[0])
if sel_name is None:
    cand = next((c for c in val_metrics.columns if c.lower() in ('model','candidate','model_name','estimator')), None)
    lower = primary.lower() in ('brier','brier_score','log_loss','mae','rmse','loss')
    if cand and primary in val_metrics.columns:
        sel_name = str(val_metrics.sort_values(primary, ascending=lower).iloc[0][cand])
    elif cand and {'metric','value'}.issubset(val_metrics.columns):
        sub = val_metrics[val_metrics['metric'].astype(str) == primary]
        if len(sub): sel_name = str(sub.sort_values('value', ascending=lower).iloc[0][cand])

model_path = inp['latest_model']
if sel_name:
    low = sel_name.lower()
    if 'logistic' in low or '07' in low: model_path = inp['07_model_train_logistic_N8YSRX76/model.pkl']
    elif 'alternative' in low or '08' in low: model_path = inp['08_model_train_alternative_M1N0MCXA/alternative_model.pkl']

def pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)
model = pkl(model_path)
pre = pkl(inp['05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl'])
sel = pkl(inp['06_features_select_features_BD9SGMPY/selector.pkl'])

def fnames(o):
    if o is None: return None
    if hasattr(o, 'feature_names_in_'): return list(o.feature_names_in_)
    if hasattr(o, 'named_steps'):
        for s in o.named_steps.values():
            ns = fnames(s)
            if ns is not None: return ns
    return None

def classes_of(o):
    if hasattr(o, 'classes_'): return np.asarray(o.classes_)
    if hasattr(o, 'named_steps'): return classes_of(o.named_steps[list(o.named_steps)[-1]])
    raise AttributeError('classes_ not found')

model_in = fnames(model)
if hasattr(model, 'named_steps') and hasattr(model, 'predict_proba'):
    est = model
elif model_in is not None and set(model_in).issubset(set(raw_cols)) and len(model_in) == len(raw_cols):
    est = model
else:
    steps = []; sel_in = fnames(sel); sel_has_pre = hasattr(sel, 'named_steps') and len(sel.named_steps) > 1
    if pre is not None and not sel_has_pre:
        need_pre = (model_in is not None and not set(model_in).issubset(set(raw_cols))) if sel is None else (sel_in is None or not set(sel_in).issubset(set(raw_cols)))
        if need_pre: steps.append(('preprocessor', pre))
    if sel is not None: steps.append(('selector', sel))
    steps.append(('model', model))
    est = Pipeline(steps) if len(steps) > 1 else model
input_cols = fnames(est) or raw_cols
if any(c in drop_cols for c in input_cols): raise ValueError('target/excluded column reached model')
X = test[input_cols]
proba = np.asarray(est.predict_proba(X))
classes = classes_of(est)
y_true = test[targets[0]].to_numpy()
thr = 0.5 if threshold is None else float(threshold)
pos = None

def bin_metric(name, yb, p, yt, yp, pos_label):
    if name == 'roc_auc': return roc_auc_score(yb, p)
    if name in ('pr_auc','average_precision'): return average_precision_score(yb, p)
    if name == 'balanced_accuracy': return balanced_accuracy_score(yt, yp)
    if name in ('brier','brier_score'): return brier_score_loss(yb, p)
    if name == 'sensitivity': return float(np.mean(yp[yb == 1] == pos_label)) if np.any(yb == 1) else np.nan
    if name == 'specificity': return float(np.mean(yp[yb == 0] != pos_label)) if np.any(yb == 0) else np.nan
    return np.nan

rng = np.random.default_rng(42); vals = []
if task.get('type') == 'binary' or len(classes) == 2:
    pos = task.get('positive_label', task.get('positive_class', classes[1]))
    if pos not in classes: pos = classes[1]
    j = list(classes).index(pos); p = proba[:, j]; yb = (y_true == pos).astype(int)
    neg = classes[0] if pos != classes[0] else classes[1]
    yp = np.where(p >= thr, pos, neg)
    metrics = {
        'roc_auc': roc_auc_score(yb, p),
        'pr_auc': average_precision_score(yb, p),
        'balanced_accuracy': balanced_accuracy_score(y_true, yp),
        'brier_score': brier_score_loss(yb, p),
        'sensitivity': bin_metric('sensitivity', yb, p, y_true, yp, pos),
        'specificity': bin_metric('specificity', yb, p, y_true, yp, pos),
    }
    if primary not in metrics: raise KeyError(f'primary {primary} not in {list(metrics)}')
    for _ in range(1000):
        idx = rng.integers(0, len(y_true), len(y_true))
        ybi, pi = yb[idx], p[idx]
        if primary in ('roc_auc','pr_auc','average_precision') and len(np.unique(ybi)) < 2: continue
        v = bin_metric(primary, ybi, pi, y_true[idx], yp[idx], pos)
        if np.isfinite(v): vals.append(v)
else:
    present = [c for c in classes if np.any(y_true == c)]
    def multi_primary(yt, pr):
        if primary in ('roc_auc_ovr','roc_auc'):
            if len([c for c in classes if np.any(yt == c)]) == len(classes):
                return roc_auc_score(yt, pr, multi_class='ovr', average='macro', labels=classes)
            vv = [roc_auc_score((yt == c).astype(int), pr[:, list(classes).index(c)]) for c in classes if np.any(yt == c) and len(np.unique(yt == c)) > 1]
            return float(np.mean(vv)) if vv else np.nan
        if primary in ('macro_f1','f1_macro'):
            return f1_score(yt, classes[np.argmax(pr, axis=1)], average='macro', labels=classes)
        raise KeyError(primary)
    auc = multi_primary(y_true, proba)
    yp = classes[np.argmax(proba, axis=1)]
    f1m = f1_score(y_true, yp, average='macro', labels=classes)
    metrics = {'roc_auc_ovr': auc, 'roc_auc': auc, 'macro_f1': f1m, 'f1_macro': f1m}
    if primary not in metrics: raise KeyError(primary)
    if len(present) < len(classes): print('classes left out:', [c for c in classes if c not in present][:5])
    for _ in range(1000):
        idx = rng.integers(0, len(y_true), len(y_true))
        v = multi_primary(y_true[idx], proba[idx])
        if np.isfinite(v): vals.append(v)

ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

def sig(x): return float(f"{x:.4g}") if np.isfinite(x) else np.nan
rows = [{'metric': k, 'value': sig(v), 'ci_low': sig(ci[0]) if k == primary else np.nan, 'ci_high': sig(ci[1]) if k == primary else np.nan} for k, v in metrics.items()]
mlca_runtime.save_table(pd.DataFrame(rows), 'test_metrics.csv', 'Test metrics')
mlca_runtime.record_metric(primary, float(metrics[primary]), 'test', ci=ci)
if primary == 'roc_auc': mlca_runtime.record_fact_candidate('test_roc_auc', float(metrics['roc_auc']), source='test_metrics.csv')
print('test rows', len(test), 'features', len(input_cols))
print('model', model_path.name, 'threshold', thr, 'positive', pos)
print(primary, sig(metrics[primary]), 'CI', sig(ci[0]), sig(ci[1]))
