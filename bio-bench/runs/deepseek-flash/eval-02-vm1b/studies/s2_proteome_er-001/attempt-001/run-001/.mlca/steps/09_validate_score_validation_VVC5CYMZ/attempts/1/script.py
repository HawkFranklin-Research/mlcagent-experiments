import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

targets = list(protocol['targets']); target = targets[0]
excluded = set(protocol.get('excluded_columns') or [])
group_col = protocol.get('group_column')
primary = protocol['task']['primary_metric']
pos_label_decl = protocol['task'].get('positive_label') or protocol['task'].get('positive_class')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

all_cols = list(train.columns)
feat_cols = [c for c in all_cols if c not in set(targets) and c not in excluded and c != group_col]
key_like = []
for c in feat_cols:
    s = train[c]
    if s.dtype == object or s.dtype.kind in 'iu':
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

def ld(k):
    with open(inp[k], 'rb') as f:
        return pickle.load(f)

prep = ld('05_preprocess_fit_preprocessing_5K21TQY3/preprocessor.pkl')
sel = ld('06_features_select_features_RDJRDVNY/selector.pkl')
m_log = ld('07_model_train_logistic_HMHZA6MZ/model.pkl')
m_alt = ld('08_model_train_alternative_R3YZHT6E/alternative_model.pkl')

def to_pipeline(model):
    if hasattr(model, 'feature_names_in_'):
        return model
    steps = []
    if prep is not None: steps.append(('prep', prep))
    if sel is not None: steps.append(('sel', sel))
    steps.append(('model', model))
    return Pipeline(steps)

y_val = np.asarray(val[target])
rows = []; results = {}
for name, model in [('logistic', m_log), ('alternative', m_alt)]:
    pipe = to_pipeline(model)
    cols = list(pipe.feature_names_in_) if hasattr(pipe, 'feature_names_in_') else feat_cols
    proba = pipe.predict_proba(val[cols])
    classes = list(pipe.classes_) if hasattr(pipe, 'classes_') else list(model.classes_)
    pos_idx = classes.index(pos_label_decl) if (pos_label_decl is not None and pos_label_decl in classes) else len(classes) - 1
    pos_label = classes[pos_idx]
    proba_pos = proba[:, pos_idx]
    y_bin = (y_val == pos_label).astype(int)
    auc = float(roc_auc_score(y_bin, proba_pos))
    ap = float(average_precision_score(y_bin, proba_pos))
    brier = float(brier_score_loss(y_bin, proba_pos))
    fpr, tpr, thr = roc_curve(y_bin, proba_pos)
    j = tpr - fpr
    fin = np.isfinite(thr)
    thr_f, j_f = thr[fin], j[fin]
    youden = float(thr_f[int(np.argmax(j_f))])
    pred = (proba_pos >= youden).astype(int)
    bacc = float(balanced_accuracy_score(y_bin, pred))
    results[name] = {'roc_auc': auc, 'pr_auc': ap, 'balanced_accuracy': bacc, 'brier': brier}
    for metric, v in [('roc_auc', auc), ('pr_auc', ap), ('balanced_accuracy', bacc), ('brier', brier)]:
        rows.append({'model': name, 'metric': metric, 'value': round(v, 6),
                     'threshold': round(youden, 6) if metric == 'balanced_accuracy' else float('nan')})

df = pd.DataFrame(rows)[['model', 'metric', 'value', 'threshold']]
mlca_runtime.save_table(df, 'validation_metrics.csv',
                        caption='Binary validation metrics per candidate; Youden J threshold from validation only.')

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_auc = float(results[best_name]['roc_auc'])
mlca_runtime.record_metric(primary, best_auc, split='val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_auc, source='validation_metrics.csv')
print(f"feat_cols={len(feat_cols)} target={target} primary={primary}")
print(f"best={best_name} val_{primary}={best_auc:.4f}")
