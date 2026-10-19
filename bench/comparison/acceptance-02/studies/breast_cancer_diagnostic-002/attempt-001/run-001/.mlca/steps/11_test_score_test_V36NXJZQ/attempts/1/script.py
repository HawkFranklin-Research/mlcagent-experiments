import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, confusion_matrix)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']; primary_metric = task['primary_metric']
targets = protocol['targets']; excluded = protocol.get('excluded_columns') or []
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train']); val = pd.read_parquet(inp['val']); test = pd.read_parquet(inp['test'])
target = targets[0]

pre = pickle.load(open(inp['06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inp['07_features_select_features_B35TTT17/selector.pkl'], 'rb'))
m1 = pickle.load(open(inp['08_model_train_logistic_71AKM31B/model.pkl'], 'rb'))
m2 = pickle.load(open(inp['09_model_train_alternative_QMPABASH/alternative_model.pkl'], 'rb'))

def feat_cols(df):
    drop = set(targets + excluded + ([group_col] if group_col else []))
    keep = []
    for c in df.columns:
        if c in drop: continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and len(s) > 10:
            continue
        keep.append(c)
    return keep

RAW = feat_cols(train)
def build(mdl):
    if hasattr(mdl, 'feature_names_in_') and list(mdl.feature_names_in_) == RAW:
        return mdl
    return Pipeline([('pre', pre), ('sel', sel), ('model', mdl)])

Xv = val[RAW]; yv = val[target]
def vscore(p):
    cls = list(p.named_steps['model'].classes_) if not hasattr(p, 'classes_') else list(p.classes_)
    pr = p.predict_proba(Xv)
    if len(cls) == 2:
        yb = (yv.values == cls[1]).astype(int)
        return roc_auc_score(yb, pr[:, 1]) if primary_metric == 'roc_auc' else balanced_accuracy_score(yb, (pr[:, 1] >= 0.5).astype(int))
    return -np.inf

p1 = build(m1); p2 = build(m2)
s1 = vscore(p1); s2 = vscore(p2)
pipe = p1 if s1 >= s2 else p2
classes = list(pipe.named_steps['model'].classes_ if hasattr(pipe, 'named_steps') else pipe.classes_)

def find_thr(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if 'threshold' in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None: return r
    return None
thresh = 0.5
for mf in ['08_model_train_logistic_71AKM31B/manifest.json',
           '09_model_train_alternative_QMPABASH/manifest.json',
           '10_validate_score_validation_6RN3B4EE/manifest.json']:
    try:
        t = find_thr(json.loads(inp[mf].read_text()))
        if t is not None: thresh = t; break
    except Exception:
        pass

Xt = test[RAW]; yt = test[target]
proba = pipe.predict_proba(Xt)

B = 1000
def bci(fn, y_t, *args):
    rng = np.random.RandomState(42); n = len(y_t); vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        try: vals.append(fn(y_t[idx], *[a[idx] for a in args]))
        except Exception: pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

rows = []
if len(classes) == 2:
    y_prob = proba[:, 1]; y_true = (yt.values == classes[1]).astype(int); y_pred = (y_prob >= thresh).astype(int)
    def sens(a, b):
        tn, fp, fn, tp = confusion_matrix(a, b, labels=[0, 1]).ravel()
        return tp / (tp + fn) if tp + fn > 0 else np.nan
    def spec(a, b):
        tn, fp, fn, tp = confusion_matrix(a, b, labels=[0, 1]).ravel()
        return tn / (tn + fp) if tn + fp > 0 else np.nan
    for name, fn, yp in [('roc_auc', roc_auc_score, y_prob), ('pr_auc', average_precision_score, y_prob),
                         ('balanced_accuracy', balanced_accuracy_score, y_pred), ('brier_score', brier_score_loss, y_prob),
                         ('sensitivity', sens, y_pred), ('specificity', spec, y_pred)]:
        v = float(fn(y_true, yp)); lo, hi = bci(fn, y_true, yp)
        rows.append({'metric': name, 'value': round(v, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
    prim_name = primary_metric
else:
    def macro_ovr(y_a, p_a):
        return float(np.mean([roc_auc_score((y_a == c).astype(int), p_a[:, j])
                              for j, c in enumerate(classes) if (y_a == c).any()]))
    y_hat = np.array(classes)[proba.argmax(axis=1)]
    for name, fn, yp in [('roc_auc_ovr_macro', macro_ovr, proba), ('macro_f1', lambda a, b: f1_score(a, b, average='macro'), y_hat)]:
        v = float(fn(yt.values, yp)); lo, hi = bci(fn, yt.values, yp)
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

print(f"test rows={len(test)} features={Xt.shape[1]} thresh={thresh:.4f}")
if len(prim_row):
    print(f"{primary_metric}={prim_row['value'].iloc[0]:.4f} [{prim_row['ci_low'].iloc[0]:.4f},{prim_row['ci_high'].iloc[0]:.4f}]")
