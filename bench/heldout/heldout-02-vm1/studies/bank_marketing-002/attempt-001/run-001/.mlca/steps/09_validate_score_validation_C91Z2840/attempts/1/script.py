import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']
pm = task['primary_metric']
targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

drop = set(targets) | excluded
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]

preprocessor = pickle.loads(inp['05_preprocess_fit_preprocessing_VSWY5WH2/preprocessor.pkl'].read_bytes())
selector = pickle.loads(inp['06_features_select_features_4XA2PFRM/selector.pkl'].read_bytes())
models = {
    'logistic':    pickle.loads(inp['07_model_train_logistic_NVNX9XZ7/model.pkl'].read_bytes()),
    'alternative': pickle.loads(inp['08_model_train_alternative_F5PXV52Q/alternative_model.pkl'].read_bytes()),
}

def build_prefix():
    if isinstance(selector, Pipeline):
        first = selector.named_steps[list(selector.named_steps)[0]]
        if isinstance(first, ColumnTransformer):
            return selector
    return Pipeline([('preprocessor', preprocessor), ('selector', selector)])

prefix = build_prefix()

def first_input_cols(obj):
    o = obj
    while isinstance(o, Pipeline):
        o = o.named_steps[list(o.named_steps)[0]]
    if hasattr(o, 'feature_names_in_'):
        return list(o.feature_names_in_)
    return feature_cols

def last_classes(model):
    o = model
    while isinstance(o, Pipeline):
        o = o.named_steps[list(o.named_steps)[-1]]
    return list(o.classes_)

def score(model):
    if isinstance(model, Pipeline):
        proba = model.predict_proba(val[first_input_cols(model)])
    else:
        Xt = prefix.transform(val[first_input_cols(prefix)])
        proba = model.predict_proba(Xt)
    classes = last_classes(model)
    pos_label = classes[1]
    y_true = val[target].values
    y_bin = (y_true == pos_label).astype(int)
    p = proba[:, 1]
    auc = roc_auc_score(y_bin, p)
    ap = average_precision_score(y_bin, p)
    fpr, tpr, thr = roc_curve(y_bin, p)
    thr_j = float(thr[int(np.argmax(tpr - fpr))])
    y_pred = (p >= thr_j).astype(int)
    ba = balanced_accuracy_score(y_bin, y_pred)
    brier = brier_score_loss(y_bin, p)
    return {'roc_auc': auc, 'pr_auc': ap, 'balanced_accuracy': ba,
            'brier': brier, 'threshold': thr_j, 'pos_label': str(pos_label)}

results = {name: score(m) for name, m in models.items()}

maximize = pm in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'f1', 'accuracy')
best_name = max(results, key=lambda n: results[n][pm]) if maximize \
            else min(results, key=lambda n: results[n][pm])

rows = []
for name, r in results.items():
    for m in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier']:
        thr = r['threshold'] if m == 'balanced_accuracy' else np.nan
        rows.append({'model': name, 'metric': m,
                     'value': round(float(r[m]), 6), 'threshold': thr})
mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv',
                        caption='Validation metrics per candidate')

mlca_runtime.record_metric(pm, float(results[best_name][pm]), 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc',
                                   float(results[best_name]['roc_auc']),
                                   source='validation_metrics.csv')

print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
print(f"best_model={best_name} {pm}={results[best_name][pm]:.4f}")
print(f"best_val_roc_auc={results[best_name]['roc_auc']:.4f}")
