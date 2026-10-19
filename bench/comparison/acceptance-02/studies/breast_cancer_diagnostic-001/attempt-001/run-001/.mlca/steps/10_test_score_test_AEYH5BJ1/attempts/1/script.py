import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, recall_score, f1_score, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
if 'targets' not in proto or 'task' not in proto:
    raise KeyError(f'_protocol missing targets/task; available top-level keys: {list(proto)}')
task = proto['task']
if 'primary_metric' not in task:
    raise KeyError(f'_protocol task missing primary_metric; available task keys: {list(task)}')
tgt = proto['targets'][0]
primary = task['primary_metric']
ttype = task.get('type', 'binary')
excl = list(proto.get('excluded_columns', []))
grp = proto.get('group_column')

test = pd.read_parquet(inp['test'])
feature_cols = []
for c in test.columns:
    if c == tgt or c in excl or c == grp:
        continue
    s = test[c]
    if s.dtype == object or str(s.dtype).startswith('string'):
        if s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

def cols_for(pipe, df):
    cols = feature_cols
    if hasattr(pipe, 'feature_names_in_'):
        exp = list(pipe.feature_names_in_)
        missing = [c for c in exp if c not in df.columns]
        if missing:
            raise ValueError(f'model expects missing columns: {missing}')
        cols = exp
    return df[cols]

val = pd.read_parquet(inp['val'])
yv = val[tgt].reset_index(drop=True)
Xv = val[feature_cols].reset_index(drop=True)

def load(p):
    return pickle.load(open(p, 'rb'))

def as_pipe(m):
    if isinstance(m, Pipeline):
        return m
    pre = load(inp['05_preprocess_fit_preprocessing_6Y57Q816/preprocessor.pkl'])
    sel = load(inp['06_features_select_features_HZT7Z0Q4/selector.pkl'])
    if isinstance(sel, Pipeline) and hasattr(sel, 'feature_names_in_') and list(sel.feature_names_in_) == feature_cols:
        return Pipeline([('selector', sel), ('model', m)])
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', m)])

models = [
    ('logistic', as_pipe(load(inp['07_model_train_logistic_94MG7111/model.pkl']))),
    ('alternative', as_pipe(load(inp['08_model_train_alternative_227Y318V/alternative_model.pkl']))),
]

def score(pipe):
    p = pipe.predict_proba(cols_for(pipe, Xv))
    cls = list(pipe.classes_)
    pm = primary.lower()
    if ttype == 'binary':
        pl = task.get('positive_label', cls[-1] if len(cls) > 1 else cls[0])
        if pl not in cls:
            raise ValueError(f'positive label {pl!r} not in model classes {cls}')
        j = cls.index(pl)
        yb = (yv == pl).astype(int)
        pr = p[:, j]
        if 'roc_auc' in pm or pm == 'auc':
            return roc_auc_score(yb, pr)
        if 'pr_auc' in pm or 'average_precision' in pm:
            return average_precision_score(yb, pr)
        if 'brier' in pm:
            return brier_score_loss(yb, pr)
        if 'balanced_accuracy' in pm:
            return balanced_accuracy_score(yb, (pr >= 0.5).astype(int))
        return roc_auc_score(yb, pr)
    present = [c for c in cls if c in set(yv)]
    if 'f1' in pm:
        return f1_score(yv, pipe.predict(cols_for(pipe, Xv)), average='macro', labels=cls)
    vals = [roc_auc_score((yv == c).astype(int), p[:, cls.index(c)]) for c in present]
    return float(np.mean(vals)) if vals else np.nan

scores = [score(m) for _, m in models]
higher = not any(w in primary.lower() for w in ['brier', 'loss', 'error'])
best_idx = int(np.nanargmax(scores) if higher else np.nanargmin(scores))
best = models[best_idx][1]

threshold = None
vm = json.loads(inp['09_validate_score_validation_D7XNK4JS/manifest.json'].read_text())
def find(o, needle):
    if isinstance(o, dict):
        for k, v in o.items():
            if needle in str(k).lower():
                return v
            r = find(v, needle)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, needle)
            if r is not None:
                return r
    return None
thr_obj = find(vm, 'threshold')
if thr_obj is not None:
    try:
        threshold = float(thr_obj)
    except Exception:
        pass
if ttype == 'binary' and threshold is None:
    pv = best.predict_proba(cols_for(best, Xv))
    cls = list(best.classes_)
    pl = task.get('positive_label', cls[-1])
    j = cls.index(pl)
    fpr, tpr, thr = roc_curve((yv == pl).astype(int), pv[:, j])
    threshold = float(thr[int(np.argmax(tpr - fpr))])

X = test[feature_cols].reset_index(drop=True)
pt = best.predict_proba(cols_for(best, test))
cls = list(best.classes_)

def bci(y, proba, pred, B=1000):
    rng = np.random.default_rng(42)
    n = len(y)
    names = (['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score', 'sensitivity', 'specificity']
             if ttype == 'binary' else ['roc_auc_ovr', 'f1_macro'])
    vals = {k: [] for k in names}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb, pr = y[idx], proba[idx], pred[idx]
        if ttype == 'binary':
            if len(np.unique(yb)) > 1:
                vals['roc_auc'].append(roc_auc_score(yb, pb))
                vals['pr_auc'].append(average_precision_score(yb, pb))
            vals['balanced_accuracy'].append(balanced_accuracy_score(yb, pr))
            vals['brier_score'].append(brier_score_loss(yb, pb))
            vals['sensitivity'].append(recall_score(yb, pr, pos_label=1, zero_division=0))
            vals['specificity'].append(recall_score(yb, pr, pos_label=0, zero_division=0))
        else:
            present = [c for c in cls if c in set(yb)]
            arr = []
            for c in present:
                j = cls.index(c)
                arr.append(roc_auc_score((yb == c).astype(int), pb[:, j]))
            if arr:
                vals['roc_auc_ovr'].append(float(np.mean(arr)))
            vals['f1_macro'].append(f1_score(yb, np.array(cls)[pr], average='macro', labels=cls))
    out = {}
    for k, v in vals.items():
        a = np.array([x for x in v if np.isfinite(x)])
        out[k] = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)
    return out

if ttype == 'binary':
    pl = task.get('positive_label', cls[-1] if len(cls) > 1 else cls[0])
    if pl not in cls:
        raise ValueError(f'positive label {pl!r} not in model classes {cls}')
    j = cls.index(pl)
    proba = pt[:, j]
    yb = (test[tgt] == pl).astype(int).reset_index(drop=True)
    pred = (proba >= threshold).astype(int)
    metrics = {
        'roc_auc': roc_auc_score(yb, proba),
        'pr_auc': average_precision_score(yb, proba),
        'balanced_accuracy': balanced_accuracy_score(yb, pred),
        'brier_score': brier_score_loss(yb, proba),
        'sensitivity': recall_score(yb, pred, pos_label=1, zero_division=0),
        'specificity': recall_score(yb, pred, pos_label=0, zero_division=0),
    }
    ci = bci(yb, proba, pred)
else:
    pred_idx = np.argmax(pt, axis=1)
    present = [c for c in cls if c in set(test[tgt])]
    arr = [roc_auc_score((test[tgt] == c).astype(int), pt[:, cls.index(c)]) for c in present]
    metrics = {
        'roc_auc_ovr': float(np.mean(arr)) if arr else np.nan,
        'f1_macro': f1_score(test[tgt], np.array(cls)[pred_idx], average='macro', labels=cls),
    }
    ci = bci(test[tgt].reset_index(drop=True), pt, pred_idx)

def sig4(v):
    return float(f'{v:.4g}') if np.isfinite(v) else np.nan

rows = []
for k, v in metrics.items():
    lo, hi = ci.get(k, (np.nan, np.nan))
    rows.append({'metric': k, 'value': sig4(v), 'ci_low': sig4(lo), 'ci_high': sig4(hi)})
df = pd.DataFrame(rows)
mlca_runtime.save_table(df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

if primary not in metrics:
    raise KeyError(f'primary metric {primary!r} not computed; available metrics: {list(metrics)}')
mlca_runtime.record_metric(primary, float(metrics[primary]), split='test', ci=ci[primary])
if 'roc_auc' in metrics:
    mlca_runtime.record_fact_candidate('test_roc_auc', float(metrics['roc_auc']), source='test_metrics.csv')

print(f'test rows {len(test)}; features {len(feature_cols)}; model {models[best_idx][0]}')
print(f'{primary}={metrics[primary]:.4f} CI {ci[primary][0]:.4f}-{ci[primary][1]:.4f}')
if ttype == 'binary':
    print(f'threshold={threshold:.4f}; pos_label={pl!r}')
else:
    left = [c for c in cls if c not in set(test[tgt])]
    if left:
        print(f'classes left out: {left}')
