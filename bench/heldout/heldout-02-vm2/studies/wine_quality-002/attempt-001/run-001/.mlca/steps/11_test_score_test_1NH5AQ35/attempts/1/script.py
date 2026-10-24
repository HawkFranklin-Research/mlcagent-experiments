import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

proto = json.loads(inp['_protocol'].read_text())
task = proto['task']; targets = proto['targets']
excluded = proto.get('excluded_columns', []) or []
group = proto.get('group_column')
primary = task['primary_metric']
tgt = targets[0]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

models = {
    'logistic': load(inp['08_model_train_logistic_DZS6CMQH/model.pkl']),
    'alternative': load(inp['09_model_train_alternative_SHFBCB82/alternative_model.pkl']),
}
pre = load(inp['06_preprocess_fit_preprocessing_HTY1VTC0/preprocessor.pkl'])
sel = load(inp['07_features_select_features_Z274ACYD/selector.pkl'])

def get_pred(est, df):
    if hasattr(est, 'named_steps'):
        cols = list(est.feature_names_in_)
        return est.predict_proba(df[cols]), est.predict(df[cols]), list(est.classes_)
    drop = [c for c in targets + excluded + ([group] if group else []) if c in df.columns]
    X = df.drop(columns=drop, errors='ignore')
    Z = sel.transform(pre.transform(X))
    return est.predict_proba(Z), est.predict(Z), list(est.classes_)

def ova_auc(y, p, classes):
    s = []
    for j, c in enumerate(classes):
        n_pos = int((y == c).sum())
        if n_pos == 0 or n_pos == len(y):
            continue
        s.append(roc_auc_score((y == c).astype(int), p[:, j]))
    return float(np.mean(s)) if s else float('nan')

val = pd.read_parquet(inp['val'])
y_val = val[tgt].values
val_scores = {}
for name, m in models.items():
    try:
        p, _, cls = get_pred(m, val)
        val_scores[name] = ova_auc(y_val, p, list(cls))
    except Exception:
        val_scores[name] = -np.inf
best = max(val_scores, key=val_scores.get)
model = models[best]

test = pd.read_parquet(inp['test'])
y_test = test[tgt].values
proba, y_pred, classes = get_pred(model, test)
classes = list(classes)

full = {
    'roc_auc_ovr': ova_auc(y_test, proba, classes),
    'f1_macro': float(f1_score(y_test, y_pred, average='macro', labels=classes, zero_division=0)),
}

rng = np.random.RandomState(42); n = len(y_test)
boot = {'roc_auc_ovr': [], 'f1_macro': []}
for _ in range(1000):
    idx = rng.randint(0, n, n)
    yy = y_test[idx]; pp = proba[idx]; yp = y_pred[idx]
    boot['roc_auc_ovr'].append(ova_auc(yy, pp, classes))
    boot['f1_macro'].append(float(f1_score(yy, yp, average='macro', labels=classes, zero_division=0)))

rows = []
for k, v in full.items():
    arr = np.asarray(boot[k], dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    rows.append({'metric': k, 'value': round(float(v), 4),
                 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4)})
    mlca_runtime.record_metric(k, float(v), 'test', ci=(float(lo), float(hi)))
mlca_runtime.save_table(pd.DataFrame(rows), 'test_metrics.csv')
mlca_runtime.record_fact_candidate('test_roc_auc', float(full['roc_auc_ovr']), source='test_metrics.csv')

pred_df = pd.DataFrame(index=test.index)
pred_df['row_id'] = test.index
pred_df['partition'] = 'test'
pred_df[f'y_true__{tgt}'] = y_test
pred_df[f'y_pred__{tgt}'] = y_pred
for j, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"selected={best} val_score={val_scores[best]:.4f} n_test={n}")
print(f"test roc_auc_ovr={full['roc_auc_ovr']:.4f} f1_macro={full['f1_macro']:.4f}")
