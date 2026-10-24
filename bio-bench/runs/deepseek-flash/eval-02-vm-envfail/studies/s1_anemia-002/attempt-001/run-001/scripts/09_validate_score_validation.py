import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, roc_curve,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
task = proto['task']
targets = proto['targets']
excluded = proto['excluded_columns']
gc = proto.get('group_column')
primary = task['primary_metric']
n_classes = int(task.get('n_classes', 2))

val = pd.read_parquet(inp['val'])

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

candidates = {
    'logistic': load_pkl(inp['07_model_train_logistic_SGR96899/model.pkl']),
    'alternative': load_pkl(inp['08_model_train_alternative_71A7RSCC/alternative_model.pkl']),
}

def feats_of(m, df):
    fni = getattr(m, 'feature_names_in_', None)
    if fni is not None:
        return df[list(fni)]
    cols = [c for c in df.columns if c not in targets and c not in excluded]
    if gc:
        cols = [c for c in cols if c != gc]
    return df[cols]

y_val_raw = val[targets[0]].values

def eval_model(m):
    Xv = feats_of(m, val)
    proba = m.predict_proba(Xv)
    classes = list(m.classes_)
    if n_classes == 2:
        pos_label = classes[1]
        y_bin = (y_val_raw == pos_label).astype(int)
        p = proba[:, 1]
        auc = float(roc_auc_score(y_bin, p))
        pr = float(average_precision_score(y_bin, p))
        brier = float(np.mean((y_bin - p) ** 2))
        fpr, tpr, thr = roc_curve(y_bin, p)
        yj = float(thr[int(np.argmax(tpr - fpr))])
        preds = (p >= yj).astype(int)
        ba = float(balanced_accuracy_score(y_bin, preds))
        return {'roc_auc': auc, 'pr_auc': pr,
                'balanced_accuracy': ba, 'brier': brier}, yj
    y_idx = np.array([classes.index(v) for v in y_val_raw])
    aucs = []
    for j in range(len(classes)):
        yb = (y_idx == j).astype(int)
        if 0 < yb.sum() < len(yb):
            aucs.append(roc_auc_score(yb, proba[:, j]))
    macro_auc = float(np.mean(aucs)) if aucs else float('nan')
    preds = m.predict(Xv)
    f1 = float(f1_score(y_val_raw, preds, average='macro'))
    return {'roc_auc': macro_auc, 'macro_f1': f1}, float('nan')

rows, results = [], {}
for name, m in candidates.items():
    met, thr = eval_model(m)
    results[name] = met
    for k, v in met.items():
        rows.append({'model': name, 'metric': k, 'value': float(v), 'threshold': thr})

df_metrics = pd.DataFrame(rows)
mlca_runtime.save_table(
    df_metrics, 'validation_metrics.csv',
    caption=f'Validation metrics per candidate model (primary: {primary})',
)

lower_is_better = primary in {'brier', 'brier_score', 'log_loss', 'rmse', 'mae'}
best, best_val = None, None
for name, met in results.items():
    v = met.get(primary)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        continue
    if best is None or (v < best_val if lower_is_better else v > best_val):
        best, best_val = name, v

mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
mlca_runtime.record_metric(primary, float(best_val), 'val')
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_val),
                                       source='validation_metrics.csv')

print('candidates:', list(candidates.keys()))
print('primary:', primary, '| best:', best, '=', round(float(best_val), 4))
print('val rows:', len(val), '| metric rows:', len(df_metrics))
