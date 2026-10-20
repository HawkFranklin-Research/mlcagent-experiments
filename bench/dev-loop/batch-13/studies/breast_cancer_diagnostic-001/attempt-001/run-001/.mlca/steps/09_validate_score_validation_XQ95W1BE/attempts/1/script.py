import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
    average_precision_score, brier_score_loss, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ['task', 'targets', 'excluded_columns']:
    if k not in proto:
        raise KeyError(f"missing {k}; top-level keys: {list(proto.keys())}")
task = proto['task']
if 'primary_metric' not in task:
    raise KeyError(f"missing task.primary_metric; task keys: {list(task.keys())}")
primary = task['primary_metric']
ttype = task.get('type', 'binary')
targets = proto['targets']
excluded = proto['excluded_columns']
group_col = proto.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]
yval = val[targets[0]].values

model_keys = [
    ('logistic', '07_model_train_logistic_QHAMT8ZK/model.pkl'),
    ('alternative', '08_model_train_alternative_SCMRPCK3/alternative_model.pkl'),
]

rows = []
primary_scores = {}
for name, k in model_keys:
    with open(inp[k], 'rb') as f:
        m = pickle.load(f)
    fn = getattr(m, 'feature_names_in_', None)
    cols = list(fn) if fn is not None else feature_cols
    proba = m.predict_proba(val[cols])
    classes = m.classes_
    if ttype == 'binary':
        pos = classes[1]
        j = int(np.where(classes == pos)[0][0])
        p = proba[:, j]
        ybin = (yval == pos).astype(int)
        roc = float(roc_auc_score(ybin, p))
        fpr, tpr, thr = roc_curve(ybin, p)
        best_thr = float(thr[int(np.argmax(tpr - fpr))])
        yhat = (p >= best_thr).astype(int)
        vals = {
            'roc_auc': roc,
            'pr_auc': float(average_precision_score(ybin, p)),
            'balanced_accuracy': float(balanced_accuracy_score(ybin, yhat)),
            'brier': float(brier_score_loss(ybin, p)),
        }
        for mname, v in vals.items():
            th = best_thr if mname == 'balanced_accuracy' else np.nan
            rows.append({'model': name, 'metric': mname, 'value': round(v, 6),
                         'threshold': th})
        print(f"{name}: roc_auc={roc:.4f} pos_label={pos} youden_thr={best_thr:.4f}")
    else:
        f1 = float(f1_score(yval, classes[np.argmax(proba, 1)], average='macro'))
        try:
            auc = float(roc_auc_score(yval, proba, multi_class='ovr',
                                      average='macro', labels=classes))
        except ValueError:
            present = [jj for jj, c in enumerate(classes) if c in set(yval)]
            auc = float(np.mean([roc_auc_score((yval == classes[jj]).astype(int),
                                               proba[:, jj]) for jj in present]))
        vals = {'roc_auc_ovr': auc, 'f1_macro': f1}
        for mname, v in vals.items():
            rows.append({'model': name, 'metric': mname, 'value': round(v, 6),
                         'threshold': np.nan})
        print(f"{name}: roc_auc_ovr={auc:.4f}")
    primary_scores[name] = round(vals[primary], 6)

maximize = primary not in {'brier'}
best_model = (max if maximize else min)(primary_scores, key=primary_scores.get)

mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv',
    caption='Validation metrics per candidate model (primary=%s)' % primary)

for name, v in primary_scores.items():
    mlca_runtime.record_metric(f"{primary}__{name}", float(v), 'val')
mlca_runtime.record_metric(primary, float(primary_scores[best_model]), 'val')
mlca_runtime.record_fact_candidate('best_model', best_model,
    source='validation_metrics.csv')
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('best_val_roc_auc',
        float(primary_scores[best_model]), source='validation_metrics.csv')

print(f"primary={primary} best={best_model} val={primary_scores[best_model]:.4f}")
print(f"train={train.shape} val={val.shape}")
