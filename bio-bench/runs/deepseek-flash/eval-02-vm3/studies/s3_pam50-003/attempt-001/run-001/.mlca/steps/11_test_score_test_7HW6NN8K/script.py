import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
task = protocol['task']
targets = list(protocol['targets'])
primary_metric = task['primary_metric']
target = targets[0]

train = pd.read_parquet(paths['train']); val = pd.read_parquet(paths['val'])
test = pd.read_parquet(paths['test']); ext = pd.read_parquet(paths['external'])

drop = set(targets) | set(protocol['excluded_columns']) | {protocol.get('group_column')}
drop.discard(None)
feature_cols = [c for c in train.columns if c not in drop and not (
    (pd.api.types.is_object_dtype(train[c]) and train[c].nunique() == len(train)) or
    (pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train)))]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

prep = load(paths['06_preprocess_fit_preprocessing_6C1GR31N/preprocessor.pkl'])
sel = load(paths['07_features_select_features_2XVWGVRB/selector.pkl'])
m_log = load(paths['08_model_train_logistic_7JTSZ4DZ/model.pkl'])
m_alt = load(paths['09_model_train_alternative_94QWT4VC/alternative_model.pkl'])

def build(prep, sel, mdl):
    if isinstance(mdl, Pipeline) and len(mdl.named_steps) >= 2:
        return mdl
    if isinstance(sel, Pipeline) and len(sel.named_steps) >= 2:
        return Pipeline(list(sel.steps) + [('model', mdl)])
    ps = list(prep.steps) if isinstance(prep, Pipeline) else [('preprocess', prep)]
    ss = list(sel.steps) if isinstance(sel, Pipeline) else [('selector', sel)]
    return Pipeline(ps + ss + [('model', mdl)])

def in_cols(pipe, fb):
    if hasattr(pipe, 'feature_names_in_'):
        return list(pipe.feature_names_in_)
    first = list(pipe.named_steps.values())[0]
    return list(getattr(first, 'feature_names_in_', fb))

def classes_of(pipe):
    if hasattr(pipe, 'classes_'):
        return list(pipe.classes_)
    return list(list(pipe.named_steps.values())[-1].classes_)

def macro_auc(y, proba, classes):
    present = [c for c in classes if (y == c).sum() > 0]
    if not present:
        return float('nan')
    return float(np.mean([roc_auc_score((y == c).astype(int), proba[:, classes.index(c)]) for c in present]))

pipes = {'logistic': build(prep, sel, m_log), 'alternative': build(prep, sel, m_alt)}
val_scores = {}
for name, p in pipes.items():
    c = in_cols(p, feature_cols)
    val_scores[name] = macro_auc(val[target].values, p.predict_proba(val[c]), classes_of(p))
valid = {k: v for k, v in val_scores.items() if not np.isnan(v)}
chosen_name = max(valid, key=valid.get) if valid else 'logistic'
chosen = pipes[chosen_name]
cols = in_cols(chosen, feature_cols)
cls = classes_of(chosen)

ytest = test[target].values
proba = chosen.predict_proba(test[cols])
pred = np.array(cls)[proba.argmax(1)]
auc = macro_auc(ytest, proba, cls)
f1 = f1_score(ytest, pred, average='macro', labels=cls, zero_division=0)

def boot(y, proba, pred, cls, kind, n=1000):
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        try:
            if kind == 'auc':
                v = macro_auc(yb, proba[idx], cls)
            else:
                v = f1_score(yb, pred[idx], average='macro', labels=cls, zero_division=0)
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float('nan'), float('nan'))

auc_ci = boot(ytest, proba, pred, cls, 'auc')
f1_ci = boot(ytest, proba, pred, cls, 'f1')

rows = [
    {'metric': primary_metric, 'value': auc, 'ci_low': auc_ci[0], 'ci_high': auc_ci[1]},
    {'metric': 'macro_f1', 'value': f1, 'ci_low': f1_ci[0], 'ci_high': f1_ci[1]},
]
for c in cls:
    rows.append({'metric': f'precision_{c}', 'value': float(precision_score((ytest == c).astype(int), (pred == c).astype(int), zero_division=0)), 'ci_low': np.nan, 'ci_high': np.nan})
    rows.append({'metric': f'recall_{c}', 'value': float(recall_score((ytest == c).astype(int), (pred == c).astype(int), zero_division=0)), 'ci_low': np.nan, 'ci_high': np.nan})

mdf = pd.DataFrame(rows)
for col in ['value', 'ci_low', 'ci_high']:
    mdf[col] = mdf[col].astype(float).round(4)
mlca_runtime.save_table(mdf, 'test_metrics.csv', caption='Test set metrics')

mlca_runtime.record_metric(primary_metric, float(auc), split='test', ci=[auc_ci[0], auc_ci[1]])
mlca_runtime.record_metric('macro_f1', float(f1), split='test', ci=[f1_ci[0], f1_ci[1]])
mlca_runtime.record_fact_candidate('test_roc_auc', float(auc), source='test_metrics.csv')

pdf = pd.DataFrame({'row_id': test.index, 'partition': 'test',
                    f'y_true__{target}': ytest, f'y_pred__{target}': pred})
for j, c in enumerate(cls):
    pdf[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pdf, 'test_predictions.parquet')

yext = ext[target].values
proba_ext = chosen.predict_proba(ext[cols])
pred_ext = np.array(cls)[proba_ext.argmax(1)]
ext_auc = macro_auc(yext, proba_ext, cls)
ext_f1 = f1_score(yext, pred_ext, average='macro', labels=cls, zero_division=0)
mlca_runtime.record_metric(primary_metric, float(ext_auc), split='external')
mlca_runtime.record_metric('macro_f1', float(ext_f1), split='external')

val_str = {k: round(v, 4) for k, v in val_scores.items()}
print(f"selected={chosen_name} val_scores={val_str}")
print(f"test {primary_metric}={auc:.4f} [{auc_ci[0]:.4f},{auc_ci[1]:.4f}] macro_f1={f1:.4f}")
print(f"external {primary_metric}={ext_auc:.4f} macro_f1={ext_f1:.4f}")
print(f"n_test={len(test)} n_ext={len(ext)} n_feat={len(cols)}")
