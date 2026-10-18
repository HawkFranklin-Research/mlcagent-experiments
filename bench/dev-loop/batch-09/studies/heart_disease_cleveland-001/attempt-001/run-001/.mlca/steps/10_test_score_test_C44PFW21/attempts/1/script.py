import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

proto = json.loads(inp['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}. Available: {list(proto.keys())}")
target = proto['targets'][0]
excluded = list(proto['excluded_columns'])
task = proto['task']
primary = task['primary_metric']
group_col = proto.get('group_column')

train = pd.read_parquet(inp['train'])
test = pd.read_parquet(inp['test'])

# feature columns per RL-0007 (excluding target, excluded, group, key-like)
drop = set(excluded) | set(proto['targets'])
if group_col: drop.add(group_col)
keep = []
for c in train.columns:
    if c in drop: continue
    s = train[c]
    n = s.nunique(dropna=False)
    if s.dtype == object and n == len(train): continue
    if pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype) and n == len(train): continue
    keep.append(c)

# pick best model by validation primary metric
vmp = pd.read_parquet(inp['09_validate_score_validation_AX923FJT/validation_metrics.parquet'])
lower_better = any(t in primary.lower() for t in ['loss', 'brier', 'mse', 'mae', 'error'])
if {'model', 'metric'}.issubset(vmp.columns):
    best_name = vmp[vmp['metric'] == primary].sort_values('value', ascending=lower_better).iloc[0]['model']
else:
    best_name = vmp.sort_values(primary, ascending=lower_better).iloc[0]['model']
if 'logistic' in str(best_name).lower() or '07' in str(best_name):
    mpath = inp['07_model_train_logistic_J5YKBG0Y/model.pkl']
else:
    mpath = inp['08_model_train_alternative_P5WQW45K/alternative_model.pkl']

mdl = pickle.loads(mpath.read_bytes())
if not (isinstance(mdl, Pipeline) and len(mdl.named_steps) > 1):
    from sklearn.compose import ColumnTransformer
    pre = pickle.loads(inp['05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl'].read_bytes())
    sel = pickle.loads(inp['06_features_select_features_CZZQV6ME/selector.pkl'].read_bytes())
    def has_pre(o):
        if isinstance(o, ColumnTransformer): return True
        if isinstance(o, Pipeline):
            return any(isinstance(v, ColumnTransformer) or 'preprocess' in k or 'column' in k for k, v in o.named_steps.items())
        return False
    def flat(o, name):
        return list(o.named_steps.items()) if isinstance(o, Pipeline) else [(name, o)]
    steps = []
    if has_pre(sel):
        steps = flat(sel, 'selector')
    elif has_pre(pre):
        steps = flat(pre, 'preprocessor') + flat(sel, 'selector')
    else:
        steps = flat(pre, 'preprocessor') + flat(sel, 'selector')
    steps.append(('model', mdl))
    mdl = Pipeline(steps)

# frozen validation threshold
threshold = 0.5
if 'metric' in vmp.columns and 'threshold' in vmp['metric'].values:
    threshold = float(vmp[vmp['metric'] == 'threshold']['value'].iloc[0])
else:
    try:
        m9 = json.loads(inp['09_validate_score_validation_AX923FJT/manifest.json'].read_text())
        for k in ['threshold', 'best_threshold', 'binary_threshold']:
            if k in m9: threshold = float(m9[k]); break
    except Exception:
        pass

cols = list(mdl.feature_names_in_) if hasattr(mdl, 'feature_names_in_') else keep
Xte = test[cols].copy()
proba = mdl.predict_proba(Xte)
classes = mdl.named_steps['model'].classes_ if isinstance(mdl, Pipeline) else mdl.classes_
y_raw = test[target].values
pos_label = task.get('positive_label', classes[1] if len(classes) == 2 else None)
if len(classes) == 2:
    pi = list(classes).index(pos_label) if pos_label in list(classes) else 1
    p_pos = np.asarray(proba[:, pi], dtype=float)
    y_bin = (y_raw == pos_label).astype(int) if pos_label is not None else (y_raw == classes[1]).astype(int)
    def calc(yb, p):
        yp = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
        return {'roc_auc': roc_auc_score(yb, p),
                'pr_auc': average_precision_score(yb, p),
                'balanced_accuracy': balanced_accuracy_score(yb, yp),
                'brier_score': brier_score_loss(yb, p),
                'sensitivity': tp / (tp + fn) if (tp + fn) else np.nan,
                'specificity': tn / (tn + fp) if (tn + fp) else np.nan}
    metrics = calc(y_bin, p_pos)
    def boot_one(idx):
        return calc(y_bin[idx], p_pos[idx])
else:
    pred = classes[proba.argmax(1)]
    metrics = {'roc_auc_ovr_macro': roc_auc_score(y_raw, proba, multi_class='ovr', average='macro', labels=classes),
               'f1_macro': f1_score(y_raw, pred, average='macro')}
    idx_all = np.arange(len(y_raw))
    def boot_one(idx):
        return {'roc_auc_ovr_macro': roc_auc_score(y_raw[idx], proba[idx], multi_class='ovr', average='macro', labels=classes),
                'f1_macro': f1_score(y_raw[idx], classes[proba[idx].argmax(1)], average='macro')}

rng = np.random.default_rng(42)
n = len(y_raw)
boot = {k: [] for k in metrics}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    try:
        vals = boot_one(idx)
        for k in boot: boot[k].append(vals.get(k, np.nan))
    except Exception:
        for k in boot: boot[k].append(np.nan)

rows = []
for k, v in metrics.items():
    arr = np.asarray(boot[k], dtype=float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    rows.append({'metric': k, 'value': float(v), 'ci_low': float(lo), 'ci_high': float(hi)})
res = pd.DataFrame(rows)
res[['value', 'ci_low', 'ci_high']] = res[['value', 'ci_low', 'ci_high']].round(4)
mlca_runtime.save_table(res, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI (1000 resamples)')

row = res[res['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(row['value']), 'test',
                           ci=(float(row['ci_low']), float(row['ci_high'])))
if primary == 'roc_auc' and 'roc_auc' in res['metric'].values:
    mlca_runtime.record_fact_candidate('test_roc_auc', float(row['value']), source='test_metrics.csv')

print(f"rows={len(res)} primary={primary}={row['value']:.4f} [{row['ci_low']:.4f},{row['ci_high']:.4f}] threshold={threshold:.3f}")
