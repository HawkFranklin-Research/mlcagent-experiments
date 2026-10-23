import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary = task['primary_metric']
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    is_id = ((s.dtype == object) or pd.api.types.is_string_dtype(s) or
             pd.api.types.is_integer_dtype(s)) and s.nunique() == len(s)
    if is_id:
        continue
    feature_cols.append(c)

target_col = targets[0]
X_val = val[feature_cols]
y_val = val[target_col]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(inputs['05_preprocess_fit_preprocessing_KCWWRC66/preprocessor.pkl'])
sel = load(inputs['06_features_retain_features_53QNTHXM/selector_all.pkl'])
models = {
    'logistic_GEE82G8E': load(inputs['07_model_train_logistic_GEE82G8E/model.pkl']),
    'alternative_QDP82960': load(inputs['08_model_train_alternative_QDP82960/alternative_model.pkl']),
}

def get_proba(model):
    cands = []
    if isinstance(model, Pipeline):
        cands.append(model)
    if isinstance(sel, Pipeline):
        cands.append(Pipeline([('frontend', sel), ('model', model)]))
    cands.append(Pipeline([('pre', pre), ('sel', sel), ('model', model)]))
    cands.append(Pipeline([('pre', pre), ('model', model)]))
    cands.append(Pipeline([('sel', sel), ('model', model)]))
    for p in cands:
        try:
            return p.predict_proba(X_val), p
        except Exception:
            continue
    raise RuntimeError(f"cannot predict {type(model)}")

def get_classes(model, pipe):
    for obj in (model, pipe):
        if isinstance(obj, Pipeline):
            last = obj.named_steps[list(obj.named_steps.keys())[-1]]
            c = getattr(last, 'classes_', None)
        else:
            c = getattr(obj, 'classes_', None)
        if c is not None:
            return list(c)
    raise RuntimeError("no classes_ found")

pos_decl = task.get('positive_label') or protocol.get('positive_label')
metrics_list = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score']
if primary not in metrics_list:
    metrics_list.append(primary)

rows = []
best_auc, best_name = -np.inf, None
for name, model in models.items():
    proba, pipe = get_proba(model)
    classes = get_classes(model, pipe)
    pos_label = pos_decl if pos_decl is not None else classes[1]
    pos_idx = classes.index(pos_label)
    p_pos = proba[:, pos_idx]
    y_bin = (y_val == pos_label).astype(int)
    auc = roc_auc_score(y_bin, p_pos)
    ap = average_precision_score(y_bin, p_pos)
    brier = brier_score_loss(y_bin, p_pos)
    fpr, tpr, thrs = roc_curve(y_bin, p_pos)
    j = tpr - fpr
    mask = np.isfinite(thrs)
    thr = float(thrs[mask][int(np.argmax(j[mask]))]) if mask.any() else 0.5
    ba = balanced_accuracy_score(y_bin, (p_pos >= thr).astype(int))
    vals = {'roc_auc': auc, 'pr_auc': ap, 'balanced_accuracy': ba, 'brier_score': brier}
    if primary not in vals:
        raise ValueError(f"primary metric {primary} unsupported")
    for m in metrics_list:
        rows.append({'model': name, 'metric': m, 'value': float(vals[m]), 'threshold': thr})
    if auc > best_auc:
        best_auc, best_name = auc, name

df = pd.DataFrame(rows)

def r4(x):
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return x
    from math import floor, log10
    return round(x, 3 - int(floor(log10(abs(x)))))

df['value'] = df['value'].apply(r4)
df['threshold'] = df['threshold'].apply(r4)

mlca_runtime.save_table(df, 'validation_metrics.csv',
                        caption='Validation metrics per candidate with Youden J thresholds.')
mlca_runtime.record_metric(primary, best_auc, 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('best_val_roc_auc', best_auc, source='validation_metrics.csv')

print('val_rows=%d feat=%d models=%d' % (len(val), len(feature_cols), len(models)))
print('best=%s %s=%.4f' % (best_name, primary, best_auc))
