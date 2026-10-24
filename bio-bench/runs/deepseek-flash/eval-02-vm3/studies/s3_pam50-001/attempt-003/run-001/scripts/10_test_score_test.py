import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp['_protocol'].read_text())
task = prot['task']; targets = prot['targets']
excluded = prot.get('excluded_columns', []); group = prot.get('group_column')
primary = task['primary_metric']; target = targets[0]

def read_pkl(k):
    with open(inp[k], 'rb') as f: return pickle.load(f)

pre = read_pkl('05_preprocess_fit_preprocessing_AP3P14P9/preprocessor.pkl')
sel = read_pkl('06_features_select_features_NBH1R9WT/selector.pkl')

try:
    vm = pd.read_parquet(inp['09_validate_score_validation_0DM6Y7TZ/validation_metrics.parquet'])
except Exception:
    vm = pd.read_csv(inp['09_validate_score_validation_0DM6Y7TZ/validation_metrics.csv'])
def col(df, names):
    for n in names:
        if n in df.columns: return n
    for c in df.columns:
        if str(c).lower() in names: return c
    return None
mc, vc, cc, sc = col(vm,['metric','metric_name','name']), col(vm,['value','score','mean']), col(vm,['model','candidate','variant','model_name','estimator','model_id']), col(vm,['split','partition'])
cand = None
if mc and vc:
    m = vm[vm[mc].astype(str).str.lower() == primary.lower()]
    if sc: m = m[m[sc].astype(str).str.lower().isin(['val','validation','valid'])]
    if len(m):
        lower = any(t in primary.lower() for t in ['brier','logloss','mse','mae'])
        row = m.loc[m[vc].idxmin() if lower else m[vc].idxmax()]
        if cc: cand = str(row[cc])
model_key = 'latest_model'
if cand:
    c = cand.lower()
    if any(t in c for t in ['08','y5m9','alt']): model_key = '08_model_train_alternative_Y5M9F5DT/alternative_model.pkl'
    elif any(t in c for t in ['07','bs7','log']): model_key = '07_model_train_logistic_BS7D3474/model.pkl'
model = read_pkl(model_key)

train = pd.read_parquet(inp['train'])
raw = []
for c in train.columns:
    if c in targets or c in excluded or c == group: continue
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith('string')) and s.nunique(dropna=True) == len(s): continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == len(s): continue
    raw.append(c)

def make_X(frame, obj=None):
    cols = getattr(obj, 'feature_names_in_', None) if obj is not None else None
    if cols is not None:
        miss = [c for c in cols if c not in frame.columns]
        if miss: raise KeyError(f'missing {miss[:3]}')
        return frame.loc[:, list(cols)]
    return frame.loc[:, raw]

def classes_of(m):
    if hasattr(m, 'classes_'): return np.asarray(m.classes_)
    if isinstance(m, Pipeline): return np.asarray(m.steps[-1][1].classes_)
    raise ValueError('no classes_')
classes = classes_of(model)

def predict_proba(frame):
    try:
        return np.asarray(model.predict_proba(make_X(frame, model)))
    except Exception:
        Xt = pre.transform(make_X(frame, pre)) if pre is not None else frame[raw]
        if sel is not None:
            if isinstance(sel, Pipeline) and len(sel.steps) >= 2:
                Xt = sel.transform(make_X(frame, sel))
            else:
                if isinstance(Xt, pd.DataFrame) and getattr(sel, 'feature_names_in_', None) is not None:
                    Xt = Xt.loc[:, list(sel.feature_names_in_)]
                Xt = sel.transform(Xt)
        if hasattr(model, 'feature_names_in_') and isinstance(Xt, pd.DataFrame):
            cols = list(model.feature_names_in_)
            if all(c in Xt.columns for c in cols): Xt = Xt.loc[:, cols]
        return np.asarray(model.predict_proba(Xt))

def macro_auc(y, p):
    present = [c for c in classes if np.any(y == c)]
    if not present: return np.nan
    vals = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        vals.append(roc_auc_score((y == c).astype(int), p[:, j]))
    return float(np.mean(vals))

def macro_f1_score(y, yp):
    labs = [c for c in classes if np.any(y == c) or np.any(yp == c)]
    return float(f1_score(y, yp, average='macro', labels=labs, zero_division=0))

def boot(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed); vals = []; nrow = len(y)
    for _ in range(n):
        idx = rng.integers(0, nrow, nrow)
        try: v = fn(y[idx], p[idx])
        except Exception: continue
        if np.isfinite(v): vals.append(v)
    if len(vals) < 10: return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))

def evaluate(df, split):
    y = df[target].to_numpy(); p = predict_proba(df)
    yp = classes[np.argmax(p, axis=1)]
    auc = macro_auc(y, p); f1 = macro_f1_score(y, yp)
    auc_ci = boot(y, p, lambda a,b: macro_auc(a,b))
    f1_ci = boot(y, p, lambda a,b: macro_f1_score(a, classes[np.argmax(b, axis=1)]))
    mlca_runtime.record_metric(primary, auc, split, ci=auc_ci)
    mlca_runtime.record_metric('macro_f1', f1, split, ci=f1_ci)
    return y, p, yp, auc, f1, auc_ci, f1_ci

test = pd.read_parquet(inp['test'])
y_test, p_test, yp_test, auc_t, f1_t, auc_t_ci, f1_t_ci = evaluate(test, 'test')
if 'external' in inp:
    evaluate(pd.read_parquet(inp['external']), 'external')

mt = pd.DataFrame([
    {'metric': primary, 'value': auc_t, 'ci_low': auc_t_ci[0], 'ci_high': auc_t_ci[1]},
    {'metric': 'macro_f1', 'value': f1_t, 'ci_low': f1_t_ci[0], 'ci_high': f1_t_ci[1]},
])
mlca_runtime.save_table(mt, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

pred = pd.DataFrame({'row_id': test.index, 'partition': 'test', 'y_true': y_test, 'y_pred': yp_test})
for j, c in enumerate(classes): pred[f'proba__{c}'] = p_test[:, j]
mlca_runtime.save_row_outputs(pred, 'test_predictions.parquet')

if primary.lower() in ('roc_auc', 'roc_auc_ovr'):
    mlca_runtime.record_fact_candidate('test_roc_auc', auc_t, source='test_metrics.csv')
print(f'test rows {len(test)}; features {len(raw)}; classes {len(classes)}')
print(f'{primary} test={auc_t:.4f} CI {auc_t_ci[0]:.4f}-{auc_t_ci[1]:.4f}')
