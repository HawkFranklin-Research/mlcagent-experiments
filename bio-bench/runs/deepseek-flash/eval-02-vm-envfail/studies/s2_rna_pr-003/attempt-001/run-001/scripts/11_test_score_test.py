import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
rq = lambda d, k: d[k] if k in d else (_ for _ in ()).throw(KeyError(f"Missing {k}; available keys: {list(d)}"))

protocol = json.loads(inp['_protocol'].read_text())
targets = list(rq(protocol, 'targets'))
excluded = set(rq(protocol, 'excluded_columns'))
group = protocol['group_column']
task = rq(protocol, 'task')
primary = rq(task, 'primary_metric')
task_type = str(task.get('type', 'binary')).lower()

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])
n = len(train)
features = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == n:
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == n:
            continue
    features.append(c)

def load_pkl(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

pre = load_pkl('06_preprocess_fit_preprocessing_1X70GNWA/preprocessor.pkl')
sel = load_pkl('07_features_select_features_41RKQQBB/selector.pkl')
model_a = load_pkl('08_model_train_logistic_Z47S9AWS/model.pkl')
model_b = load_pkl('latest_model')

vm = pd.read_csv(inp['10_validate_score_validation_QB01515K/validation_metrics.csv']) if inp['10_validate_score_validation_QB01515K/validation_metrics.csv'].exists() else None
cols = {c.lower(): c for c in vm.columns} if vm is not None else {}
mcol = next((cols[k] for k in ['model', 'model_name', 'candidate', 'estimator'] if k in cols), None)
metcol = next((cols[k] for k in ['metric', 'name'] if k in cols), None)
valcol = next((cols[k] for k in ['value', 'score'] if k in cols), None)

def key_from(v):
    s = str(v).lower()
    if 'logistic' in s:
        return '08_model_train_logistic_Z47S9AWS/model.pkl'
    if 'alternative' in s or 'latest' in s:
        return 'latest_model'
    return None

def find_key(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'model' in k.lower() and ('best' in k.lower() or 'selected' in k.lower()) and isinstance(v, str):
                r = key_from(v)
                if r:
                    return r
            r = find_key(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v)
            if r:
                return r
    return None

selected_key = None
if vm is not None and mcol and metcol and valcol:
    sub = vm[vm[metcol].astype(str).str.lower() == primary.lower()]
    if len(sub):
        higher = primary.lower() not in ['brier_score', 'log_loss', 'mae', 'rmse']
        sub = sub.sort_values(valcol, ascending=not higher)
        selected_key = key_from(sub.iloc[0][mcol])
if selected_key is None and inp['10_validate_score_validation_QB01515K/manifest.json'].exists():
    selected_key = find_key(json.loads(inp['10_validate_score_validation_QB01515K/manifest.json'].read_text()))
if selected_key is None:
    selected_key = 'latest_model' if 'latest_model' in inp else '08_model_train_logistic_Z47S9AWS/model.pkl'
chosen = load_pkl(selected_key)

def accepts_raw(obj):
    names = getattr(obj, 'feature_names_in_', None)
    return names is not None and set(names).issubset(set(features))

def predict_proba_any(df):
    names = getattr(chosen, 'feature_names_in_', None)
    if names is not None and set(names).issubset(set(features)):
        try:
            return chosen.predict_proba(df[list(names)]), chosen
        except Exception:
            pass
    front = sel if accepts_raw(sel) else Pipeline([('preprocessor', pre), ('selector', sel)])
    pipe = Pipeline([('frontend', front), ('model', chosen)])
    return pipe.predict_proba(df[features]), pipe

def find_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'threshold' in k.lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_threshold(v)
            if r is not None:
                return r
    return None

thr = None
if vm is not None and metcol and valcol:
    sub = vm[vm[metcol].astype(str).str.lower().str.contains('threshold')]
    if len(sub):
        thr = float(sub.iloc[0][valcol])
if thr is None and inp['10_validate_score_validation_QB01515K/manifest.json'].exists():
    thr = find_threshold(json.loads(inp['10_validate_score_validation_QB01515K/manifest.json'].read_text()))

if task_type != 'binary':
    raise ValueError('This frozen evaluation script expects a binary task')

proba, used_model = predict_proba_any(test)
classes = list(getattr(used_model, 'classes_', getattr(chosen, 'classes_', [])))
if len(classes) != 2:
    raise ValueError('Binary evaluation requires exactly two model classes')
pos_label = task.get('positive_label', None)
if pos_label is None:
    pos_label = classes[1]
pos_idx = classes.index(pos_label)
neg_label = classes[0] if classes[1] == pos_label else classes[1]

if thr is None:
    vp, vm_model = predict_proba_any(val)
    vclasses = list(getattr(vm_model, 'classes_', classes))
    vpos = task.get('positive_label', None)
    if vpos is None:
        vpos = vclasses[1]
    vp_idx = vclasses.index(vpos)
    yv = (val[targets[0]].values == vpos).astype(int)
    if len(np.unique(yv)) > 1:
        fpr, tpr, ths = roc_curve(yv, vp[:, vp_idx])
        thr = float(ths[np.argmax(tpr - fpr)])
    else:
        thr = 0.5

y_true = test[targets[0]].values
p = proba[:, pos_idx]
y_bin = (y_true == pos_label).astype(int)
y_pred_bin = (p >= thr).astype(int)
y_pred = np.where(y_pred_bin == 1, pos_label, neg_label)

def binary_metrics(yb, pp, t):
    ypb = (pp >= t).astype(int)
    tp = np.sum((yb == 1) & (ypb == 1)); fn = np.sum((yb == 1) & (ypb == 0))
    tn = np.sum((yb == 0) & (ypb == 0)); fp = np.sum((yb == 0) & (ypb == 1))
    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    return {
        'roc_auc': roc_auc_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan,
        'pr_auc': average_precision_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan,
        'balanced_accuracy': balanced_accuracy_score(yb, ypb) if len(np.unique(yb)) > 1 else np.nan,
        'brier_score': brier_score_loss(yb, pp),
        'sensitivity': sens,
        'specificity': spec,
    }

point = binary_metrics(y_bin, p, thr)
rng = np.random.default_rng(42)
boot = {k: [] for k in point}
for _ in range(1000):
    idx = rng.integers(0, len(y_bin), len(y_bin))
    m = binary_metrics(y_bin[idx], p[idx], thr)
    for k, v in m.items():
        if np.isfinite(v):
            boot[k].append(v)
ci = {k: (np.nanpercentile(boot[k], 2.5), np.nanpercentile(boot[k], 97.5)) if boot[k] else (np.nan, np.nan) for k in point}

metrics_df = pd.DataFrame([{'metric': k, 'value': point[k], 'ci_low': ci[k][0], 'ci_high': ci[k][1]} for k in point]).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Frozen model test metrics with 95% bootstrap CI')
pval = float(point[primary]); plo, phi = float(ci[primary][0]), float(ci[primary][1])
mlca_runtime.record_metric(primary, pval, split='test', ci=(plo, phi))
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', pval, source='test_metrics.csv')

pred = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
if len(targets) == 1:
    pred['y_true'] = test[targets[0]].values
    pred['y_pred'] = y_pred
else:
    raise ValueError('Multiple targets are not supported by this frozen binary evaluation')
for i, c in enumerate(classes):
    pred[f'proba__{c}'] = proba[:, i]
mlca_runtime.save_row_outputs(pred, 'test_predictions.parquet')

print(f"model={selected_key}")
print(f"test_rows={len(test)}")
print(f"{primary}={pval:.4f} CI=({plo:.4f},{phi:.4f})")
print(f"threshold={thr:.4f}")
