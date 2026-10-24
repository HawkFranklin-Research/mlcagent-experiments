import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = protocol['targets']
target_col = targets[0] if len(targets) == 1 else (_ for _ in ()).throw(ValueError("Expected one target"))
excluded = protocol['excluded_columns']
if isinstance(excluded, dict):
    for k in ['columns', 'excluded', 'list', 'names']:
        if k in excluded and isinstance(excluded[k], list):
            excluded = excluded[k]
            break
    else:
        excluded = excluded.get('examples', [])
if not isinstance(excluded, list):
    excluded = []
group_col = protocol.get('group_column')

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train):
            continue
    feature_cols.append(c)

def load_pkl(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl('05_preprocess_fit_preprocessing_K9GV8476/preprocessor.pkl')
selector = load_pkl('06_features_select_features_BXHGCTQP/selector.pkl')
model_log = load_pkl('07_model_train_logistic_6PX29S2M/model.pkl')
model_alt = load_pkl('08_model_train_alternative_MSPMPPWQ/alternative_model.pkl')

def get_classes(m):
    if hasattr(m, 'classes_'):
        return m.classes_
    if isinstance(m, Pipeline):
        return m.steps[-1][1].classes_
    raise ValueError("No classes_")

def evaluate(model, name, df, pos_label):
    if hasattr(model, 'feature_names_in_'):
        X = df[list(model.feature_names_in_)]
        est = model
    else:
        steps = []
        if preprocessor is not None: steps.append(('preprocessor', preprocessor))
        if selector is not None: steps.append(('selector', selector))
        steps.append(('model', model))
        est = Pipeline(steps)
        cols = list(preprocessor.feature_names_in_) if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_') else feature_cols
        X = df[cols]
    proba = est.predict_proba(X)
    classes = get_classes(est)
    pos_idx = list(classes).index(pos_label)
    proba_pos = proba[:, pos_idx]
    y_true = df[target_col].values
    y_bin = (y_true == pos_label).astype(int)
    roc_auc = roc_auc_score(y_bin, proba_pos)
    pr_auc = average_precision_score(y_bin, proba_pos)
    fpr, tpr, thr = roc_curve(y_bin, proba_pos)
    best_thr = thr[np.argmax(tpr - fpr)]
    y_pred = (proba_pos >= best_thr).astype(int)
    bal_acc = balanced_accuracy_score(y_bin, y_pred)
    brier = brier_score_loss(y_bin, proba_pos)
    return dict(model=name, roc_auc=roc_auc, pr_auc=pr_auc, balanced_accuracy=bal_acc, brier=brier, threshold=best_thr)

results = []
for m, name in [(model_log, 'logistic'), (model_alt, 'alternative')]:
    classes = get_classes(m)
    pos_label = protocol.get('task', {}).get('positive_label') or protocol.get('positive_label') or classes[1]
    results.append(evaluate(m, name, val, pos_label))

rows = []
for r in results:
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier']:
        thr = r['threshold'] if metric == 'balanced_accuracy' else np.nan
        rows.append({'model': r['model'], 'metric': metric, 'value': float(f"{r[metric]:.4g}"), 'threshold': float(f"{thr:.4g}") if not np.isnan(thr) else np.nan})
df_metrics = pd.DataFrame(rows)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics per model')

primary = protocol['task']['primary_metric']
higher = primary in ['roc_auc', 'pr_auc', 'balanced_accuracy']
best = max(results, key=lambda r: r[primary]) if higher else min(results, key=lambda r: r[primary])
best_val = best[primary]
mlca_runtime.record_fact_candidate('best_model', best['model'], source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(f"{best['roc_auc']:.4g}"), source='validation_metrics.csv')
mlca_runtime.record_metric(primary, float(f"{best_val:.4g}"), 'val')

print(f"val rows={len(val)} features={len(feature_cols)}")
print(f"best={best['model']} {primary}={best_val:.4f}")
