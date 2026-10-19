import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
target = targets[0]
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
task = protocol['task']
primary = task['primary_metric']

test = pd.read_parquet(inputs['test'])
if target not in test.columns:
    raise ValueError(f'missing target {target!r}; test columns={list(test.columns)}')
y = test[target].to_numpy()

val_path = inputs['09_validate_score_validation_RSQWT8DP/validation_metrics.csv']
try:
    val = pd.read_csv(val_path)
except Exception:
    val = pd.read_parquet(inputs['09_validate_score_validation_RSQWT8DP/validation_metrics.parquet'])
label = 'latest'
cols_lower = {str(c).lower(): c for c in val.columns}
model_col = next((cols_lower[k] for k in ['model','candidate','estimator','model_name','name'] if k in cols_lower), None)
metric_col = next((cols_lower[k] for k in ['metric','metric_name'] if k in cols_lower), None)
value_col = next((cols_lower[k] for k in ['value','score','mean'] if k in cols_lower), None)
primary_col = next((c for c in val.columns if str(c).lower() == primary.lower()), None)
if primary_col is not None and model_col is not None:
    t = val[[model_col, primary_col]].dropna()
    if len(t):
        label = str(t.loc[t[primary_col].idxmax(), model_col])
elif metric_col is not None and value_col is not None:
    t = val[val[metric_col].astype(str).str.lower() == primary.lower()]
    if model_col is not None and len(t):
        g = t.groupby(model_col)[value_col].max().sort_values(ascending=False)
        if len(g):
            label = str(g.index[0])

low = label.lower()
if 'alternative' in low or '08' in low:
    model_path = inputs['08_model_train_alternative_JKAAJDG0/alternative_model.pkl']
elif 'logistic' in low or '07' in low:
    model_path = inputs['07_model_train_logistic_Z4Z606HV/model.pkl']
else:
    model_path = inputs['latest_model']

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(inputs['05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl'])
sel = load(inputs['06_features_select_features_53HVANE8/selector.pkl'])
model = load(model_path)

def names(obj):
    try:
        if hasattr(obj, 'feature_names_in_'):
            return list(obj.feature_names_in_)
    except Exception:
        pass
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            try:
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
            except Exception:
                pass
    return None

test_cols = set(test.columns)
def raw_ok(obj):
    n = names(obj)
    return n is not None and set(n).issubset(test_cols)

if isinstance(model, Pipeline) and raw_ok(model):
    pipe = model
    raw_names = names(model)
elif isinstance(sel, Pipeline) and raw_ok(sel):
    pipe = Pipeline([('frontend', sel), ('model', model)])
    raw_names = names(sel)
elif raw_ok(pre):
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)])
    raw_names = names(pre)
else:
    raise ValueError('could not identify a fitted frontend accepting raw test columns')

X_test = test.loc[:, raw_names].copy()
def classes_of(obj):
    if isinstance(obj, Pipeline):
        est = list(obj.named_steps.values())[-1]
        return list(est.classes_)
    return list(obj.classes_)

classes = classes_of(pipe)
proba = np.asarray(pipe.predict_proba(X_test))
pred = np.asarray(pipe.predict(X_test))
if proba.shape[1] != len(classes):
    raise ValueError(f'probability columns {proba.shape[1]} != classes {len(classes)}')

def macro_auc(yy, pp):
    present = [c for c in classes if np.any(yy == c)]
    vals = []
    for c in present:
        j = classes.index(c)
        try:
            vals.append(roc_auc_score((yy == c).astype(int), pp[:, j]))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan

def macro_f1(yy, ppred):
    return float(f1_score(yy, ppred, labels=classes, average='macro', zero_division=0))

auc = macro_auc(y, proba)
f1 = macro_f1(y, pred)
n = len(y)
rng = np.random.default_rng(42)
aucs, f1s = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb, prb = y[idx], proba[idx, :], pred[idx]
    a = macro_auc(yb, pb)
    if np.isfinite(a):
        aucs.append(a)
    f1s.append(macro_f1(yb, prb))

def ci(vals):
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

auc_ci = ci(aucs)
f1_ci = ci(f1s)
if primary == 'f1_macro':
    pval, pci = f1, f1_ci
else:
    pval, pci = auc, auc_ci

rows = [[primary, pval, pci[0], pci[1]]]
if primary != 'f1_macro':
    rows.append(['f1_macro', f1, f1_ci[0], f1_ci[1]])
else:
    rows.append(['roc_auc_ovr', auc, auc_ci[0], auc_ci[1]])

df = pd.DataFrame(rows, columns=['metric', 'value', 'ci_low', 'ci_high'])
def sig4(x):
    return float(f'{x:.4g}') if np.isfinite(x) else np.nan
for c in ['value', 'ci_low', 'ci_high']:
    df[c] = df[c].map(sig4)

mlca_runtime.save_table(df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')
mlca_runtime.record_metric(primary, float(pval), split='test', ci=(float(pci[0]), float(pci[1])))
mlca_runtime.record_metric('f1_macro', float(f1), split='test', ci=(float(f1_ci[0]), float(f1_ci[1])))
if primary == 'roc_auc_ovr':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(auc), source='test_metrics.csv')

missing = [c for c in classes if not np.any(y == c)]
print(f'test rows={len(test)}')
print(f'selected={label}')
print(f'features={len(raw_names)}')
print(f'{primary}={pval:.4f} ci=[{pci[0]:.4f},{pci[1]:.4f}]')
if missing:
    print(f'missing classes={missing}')
