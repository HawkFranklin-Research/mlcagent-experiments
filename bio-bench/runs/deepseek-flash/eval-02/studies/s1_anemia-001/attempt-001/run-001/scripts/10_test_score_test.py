import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, confusion_matrix

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = set(protocol.get('excluded_columns', []))
group = protocol.get('group_column')
task = protocol['task']
primary = task['primary_metric']
task_type = task.get('type', 'binary')
test = pd.read_parquet(inp['test'])
target = targets[0]

cand = [c for c in test.columns if c not in set(targets) | excluded | ({group} if group else set())]
raw_features = []
for c in cand:
    s = test[c]
    if (s.dtype == object or str(s.dtype).startswith('string') or s.dtype.name == 'category') and s.nunique(dropna=False) == len(s):
        continue
    if s.dtype.kind in 'iu' and s.nunique(dropna=False) == len(s):
        continue
    raw_features.append(c)

def load(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

def input_cols(obj):
    return list(obj.feature_names_in_) if hasattr(obj, 'feature_names_in_') else None

pre = load('05_preprocess_fit_preprocessing_PTKJZF9S/preprocessor.pkl')
sel = load('06_features_select_features_00QZZ2DE/selector.pkl')
model = load('latest_model')

def assemble(model, pre, sel, raw_features, test_cols):
    if isinstance(model, Pipeline):
        mc = input_cols(model)
        if mc is None or set(mc).issubset(set(test_cols)):
            return model, (mc or raw_features)
    mc = input_cols(model)
    if mc is not None and set(mc).issubset(set(raw_features)) and len(mc) == len(raw_features):
        return model, mc
    if isinstance(sel, Pipeline):
        return Pipeline([('selector', sel), ('model', model)]), (input_cols(sel) or raw_features)
    sc = input_cols(sel)
    if sc is not None and set(sc).issubset(set(raw_features)):
        return Pipeline([('selector', sel), ('model', model)]), sc
    pc = input_cols(pre)
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)]), (pc or raw_features)

pipe, cols = assemble(model, pre, sel, raw_features, test.columns)
X = test[cols]
proba = pipe.predict_proba(X)
classes = np.asarray(pipe.classes_)

def find_threshold(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if 'threshold' in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        for v in o.values():
            t = find_threshold(v)
            if t is not None:
                return t
    elif isinstance(o, list):
        for v in o:
            t = find_threshold(v)
            if t is not None:
                return t
    return None

threshold = find_threshold(json.loads(inp['09_validate_score_validation_FW4BPY17/manifest.json'].read_text()))
if threshold is None:
    vm = pd.read_parquet(inp['09_validate_score_validation_FW4BPY17/validation_metrics.parquet'])
    for c in vm.columns:
        if 'threshold' in c.lower():
            vals = pd.to_numeric(vm[c], errors='coerce').dropna()
            if len(vals):
                threshold = float(vals.iloc[0]); break
    if threshold is None and {'metric', 'value'}.issubset(vm.columns):
        r = vm[vm['metric'].astype(str).str.contains('threshold', case=False, na=False)]
        if len(r):
            threshold = float(pd.to_numeric(r['value'], errors='coerce').dropna().iloc[0])
threshold = 0.5 if threshold is None else threshold

y = test[target].to_numpy()
rng = np.random.default_rng(42)
n = len(y)

if task_type == 'binary':
    pos = task.get('positive_label', protocol.get('positive_label'))
    pos = classes[1] if pos is None else pos
    pi = int(np.where(classes == pos)[0][0])
    neg = classes[1 - pi]
    p = proba[:, pi]
    y_bin = (y == pos).astype(int)
    def calc(yy, pp):
        yp = np.where(pp >= threshold, pos, neg)
        yp_bin = (yp == pos).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, yp_bin, labels=[0, 1]).ravel()
        return {
            'roc_auc': roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
            'pr_auc': average_precision_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
            'balanced_accuracy': balanced_accuracy_score(yy, yp_bin),
            'brier_score': brier_score_loss(yy, pp),
            'sensitivity': tp / (tp + fn) if (tp + fn) else np.nan,
            'specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        }
    names = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score', 'sensitivity', 'specificity']
    full = calc(y_bin, p)
    boot = {k: [] for k in names}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            m = calc(y_bin[idx], p[idx])
            for k in names:
                boot[k].append(m[k])
        except Exception:
            pass
    y_pred = np.where(p >= threshold, pos, neg)
else:
    y_pred = classes[np.argmax(proba, axis=1)]
    present = [j for j, c in enumerate(classes) if np.any(y == c)]
    def macro_auc(yy, pp):
        if len(present) == len(classes):
            return roc_auc_score(yy, pp, multi_class='ovr', average='macro', labels=classes)
        return np.mean([roc_auc_score((yy == classes[j]).astype(int), pp[:, j]) for j in present])
    full = {'roc_auc': macro_auc(y, proba), 'f1_macro': f1_score(y, y_pred, average='macro', labels=classes)}
    names = ['roc_auc', 'f1_macro']
    boot = {k: [] for k in names}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            yy, pp = y[idx], proba[idx]
            boot['roc_auc'].append(macro_auc(yy, pp))
            boot['f1_macro'].append(f1_score(yy, classes[np.argmax(pp, axis=1)], average='macro', labels=classes))
        except Exception:
            pass

rows = []
for k in names:
    vals = np.asarray(boot[k], dtype=float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    rows.append({'metric': k, 'value': full[k], 'ci_low': lo, 'ci_high': hi})
metrics = pd.DataFrame(rows, columns=['metric', 'value', 'ci_low', 'ci_high'])
for c in ['value', 'ci_low', 'ci_high']:
    metrics[c] = metrics[c].map(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(metrics, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

prim_row = metrics.loc[metrics['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(prim_row['value']), 'test', ci=(float(prim_row['ci_low']), float(prim_row['ci_high'])))
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(prim_row['value']), source='test_metrics.csv')

pred = pd.DataFrame({'row_id': test.index, 'partition': 'test', f'y_true__{target}': y, f'y_pred__{target}': y_pred})
for j, c in enumerate(classes):
    pred[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred, 'test_predictions.parquet')
print(f"test rows={len(test)} features={len(cols)}")
print(f"{primary}={prim_row['value']:.4g} ci=[{prim_row['ci_low']:.4g}, {prim_row['ci_high']:.4g}]")
