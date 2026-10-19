import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, confusion_matrix)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns') or []
group_col = protocol.get('group_column')
positive_label = protocol.get('positive_label', task.get('positive_label'))

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])
target = targets[0]

def load_pkl(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

pre = load_pkl('06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl')
sel = load_pkl('07_features_select_features_B35TTT17/selector.pkl')
m1 = load_pkl('08_model_train_logistic_71AKM31B/model.pkl')
m2 = load_pkl('09_model_train_alternative_QMPABASH/alternative_model.pkl')

def feat_cols(df):
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if len(s) > 10 and s.nunique(dropna=False) == len(s):
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                continue
        keep.append(c)
    return keep

RAW = feat_cols(train)

def is_raw(obj):
    return hasattr(obj, 'feature_names_in_') and list(obj.feature_names_in_) == list(RAW)

def build(mdl):
    if is_raw(mdl):
        return mdl
    return Pipeline([('pre', pre), ('sel', sel), ('model', mdl)])

def final_est(obj):
    if isinstance(obj, Pipeline):
        return obj.named_steps[obj.steps[-1][0]]
    return obj

def classes_of(obj):
    return list(final_est(obj).classes_)

def choose_positive(cls):
    if positive_label is not None and positive_label in cls:
        return positive_label
    return cls[1]

def vscore(p):
    cls = classes_of(p)
    pr = p.predict_proba(val[RAW])
    if len(cls) == 2:
        pos = choose_positive(cls)
        pos_idx = cls.index(pos)
        yb = (val[target].values == pos).astype(int)
        if primary_metric == 'roc_auc':
            return roc_auc_score(yb, pr[:, pos_idx])
        return balanced_accuracy_score(yb, (pr[:, pos_idx] >= 0.5).astype(int))
    if primary_metric == 'roc_auc':
        return float(np.mean([roc_auc_score((val[target].values == c).astype(int), pr[:, j])
                              for j, c in enumerate(cls) if (val[target].values == c).any()]))
    return f1_score(val[target].values, np.asarray(cls)[pr.argmax(axis=1)], average='macro')

p1, p2 = build(m1), build(m2)
pipe = p1 if vscore(p1) >= vscore(p2) else p2
classes = classes_of(pipe)

def find_thr(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if 'threshold' in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(d, list):
        for x in d:
            r = find_thr(x)
            if r is not None:
                return r
    return None

thresh = None
for key in ['08_model_train_logistic_71AKM31B/manifest.json',
            '09_model_train_alternative_QMPABASH/manifest.json',
            '10_validate_score_validation_6RN3B4EE/manifest.json']:
    try:
        t = find_thr(json.loads(inp[key].read_text()))
        if t is not None:
            thresh = t
            break
    except Exception:
        pass

if thresh is None and len(classes) == 2:
    pos = choose_positive(classes)
    pos_idx = classes.index(pos)
    yv = (val[target].values == pos).astype(int)
    pv = pipe.predict_proba(val[RAW])[:, pos_idx]
    grid = np.linspace(0, 1, 101)
    thresh = max(grid, key=lambda t: balanced_accuracy_score(yv, (pv >= t).astype(int)))
thresh = 0.5 if thresh is None else thresh

Xt = test[RAW]
yt = test[target].values
proba = pipe.predict_proba(Xt)

def bci(fn, y_t, *args):
    rng = np.random.RandomState(42)
    n = len(y_t)
    vals = []
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        try:
            v = fn(y_t[idx], *[a[idx] for a in args])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def sens(y_t, y_p):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan

def spec(y_t, y_p):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

rows = []
if len(classes) == 2:
    pos = choose_positive(classes)
    pos_idx = classes.index(pos)
    y_prob = proba[:, pos_idx]
    y_true = (yt == pos).astype(int)
    y_pred = (y_prob >= thresh).astype(int)
    for name, fn, yp in [('roc_auc', roc_auc_score, y_prob),
                         ('pr_auc', average_precision_score, y_prob),
                         ('balanced_accuracy', balanced_accuracy_score, y_pred),
                         ('brier_score', brier_score_loss, y_prob),
                         ('sensitivity', sens, y_pred),
                         ('specificity', spec, y_pred)]:
        v = float(fn(y_true, yp))
        lo, hi = bci(fn, y_true, yp)
        rows.append({'metric': name, 'value': round(v, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
    prim_name = primary_metric
else:
    def macro_ovr(y_a, p_a):
        return float(np.mean([roc_auc_score((y_a == c).astype(int), p_a[:, j])
                              for j, c in enumerate(classes) if (y_a == c).any()]))
    y_hat = np.asarray(classes)[proba.argmax(axis=1)]
    for name, fn, yp in [('roc_auc_ovr_macro', macro_ovr, proba),
                         ('macro_f1', lambda a, b: f1_score(a, b, average='macro'), y_hat)]:
        v = float(fn(yt, yp))
        lo, hi = bci(fn, yt, yp)
        rows.append({'metric': name, 'value': round(v, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
    prim_name = 'roc_auc_ovr_macro' if primary_metric == 'roc_auc' else primary_metric

dfm = pd.DataFrame(rows)
mlca_runtime.save_table(dfm, 'test_metrics.csv')

prim_row = dfm[dfm['metric'] == prim_name]
if len(prim_row):
    r = prim_row.iloc[0]
    mlca_runtime.record_metric(primary_metric, float(r['value']), 'test', ci=[float(r['ci_low']), float(r['ci_high'])])
    if primary_metric == 'roc_auc' and len(classes) == 2:
        mlca_runtime.record_fact_candidate('test_roc_auc', float(r['value']), source='test_metrics.csv')

print(f"test rows={len(test)} features={Xt.shape[1]} classes={len(classes)}")
if len(prim_row):
    print(f"{primary_metric}={prim_row['value'].iloc[0]:.4f} [{prim_row['ci_low'].iloc[0]:.4f},{prim_row['ci_high'].iloc[0]:.4f}]")
