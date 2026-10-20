import json, pickle, numpy as np, pandas as pd, mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group_col = protocol.get('group_column')
primary = task['primary_metric']
target_col = targets[0] if isinstance(targets, list) else targets

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in val.columns if c not in targets and c not in excluded and c != group_col]

with open(inp['07_model_train_logistic_8CP1FPXQ/model.pkl'], 'rb') as f:
    m_log = pickle.load(f)
with open(inp['08_model_train_alternative_D655NSWD/alternative_model.pkl'], 'rb') as f:
    m_alt = pickle.load(f)

models = [('logistic', m_log), ('alternative', m_alt)]

def classes_of(m):
    if hasattr(m, 'classes_'):
        return np.asarray(m.classes_)
    if isinstance(m, Pipeline):
        return np.asarray(m.steps[-1][1].classes_)
    raise ValueError('saved model exposes no classes_')

def features_for(m):
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)
    return feature_cols

def score(m, X, y):
    proba = m.predict_proba(X)
    classes = classes_of(m)
    present = [c for c in classes if (y == c).any()]
    if len(present) == len(classes):
        auc = roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=classes)
    else:
        aucs = []
        for c in present:
            j = int(np.where(classes == c)[0][0])
            aucs.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        auc = float(np.mean(aucs))
    pred = m.predict(X)
    f1 = f1_score(y, pred, average='macro')
    return float(auc), float(f1)

y = np.asarray(val[target_col])
rows, scores = [], {}
for name, m in models:
    X = val[features_for(m)]
    auc, f1 = score(m, X, y)
    scores[name] = auc
    rows.append({'model': name, 'metric': primary, 'value': auc, 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'f1_macro', 'value': f1, 'threshold': np.nan})

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv',
                        caption='Validation metrics by model (multiclass, threshold n/a)')

best_name = max(scores, key=scores.get)
best_auc = scores[best_name]
for n, a in scores.items():
    mlca_runtime.record_metric(f'{primary}__{n}', a, split='val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_auc), source='validation_metrics.csv')

print('metric rows:', len(rows), '| val rows:', len(val))
print('best:', best_name, round(best_auc, 4))
