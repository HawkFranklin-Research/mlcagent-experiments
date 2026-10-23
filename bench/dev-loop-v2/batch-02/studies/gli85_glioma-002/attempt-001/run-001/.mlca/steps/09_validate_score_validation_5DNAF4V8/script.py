import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
positive_label = task.get('positive_label', protocol.get('positive_label', None))

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feature_cols = [c for c in train.columns if c not in drop_cols]
for c in feature_cols[:]:
    if train[c].dtype == 'object' or str(train[c].dtype).startswith('string'):
        if train[c].nunique() == len(train):
            feature_cols.remove(c)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl(inputs['05_preprocess_fit_preprocessing_M7MKT30F/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_select_features_18G3F1C0/selector.pkl'])
model_log = load_pkl(inputs['07_model_train_logistic_TAB66YEF/model.pkl'])
model_alt = load_pkl(inputs['08_model_train_alternative_N674VYMZ/alternative_model.pkl'])

candidates = {'logistic': model_log, 'alternative': model_alt}

def has_step(obj, keywords):
    if isinstance(obj, Pipeline):
        return any(any(kw in s.lower() for kw in keywords) for s in obj.named_steps)
    return False

def get_feature_names(obj):
    return list(obj.feature_names_in_) if hasattr(obj, 'feature_names_in_') else None

def build_pipeline(classifier, preprocessor, selector, raw_cols):
    fn = get_feature_names(classifier)
    if fn is not None and set(fn) == set(raw_cols):
        return classifier
    steps = []
    selector_has_pre = selector is not None and has_step(selector, ['preprocess', 'preprocessor', 'columntransformer'])
    if selector_has_pre:
        if not has_step(classifier, ['select', 'selector']):
            steps.append(('selector', selector))
    else:
        if preprocessor is not None and not has_step(classifier, ['preprocess', 'preprocessor', 'columntransformer']):
            steps.append(('preprocessor', preprocessor))
        if selector is not None and not has_step(classifier, ['select', 'selector']) and not has_step(preprocessor, ['select', 'selector']):
            steps.append(('selector', selector))
    steps.append(('classifier', classifier))
    return Pipeline(steps)

results = []
for name, clf in candidates.items():
    pipe = build_pipeline(clf, preprocessor, selector, feature_cols)
    first_step = pipe.steps[0][1]
    input_cols = get_feature_names(first_step) or feature_cols
    X_val = val[input_cols].copy()
    y_val = val[targets[0]].copy()
    proba = pipe.predict_proba(X_val)
    classes = pipe.steps[-1][1].classes_
    pos_idx = list(classes).index(positive_label) if positive_label is not None else (1 if len(classes) == 2 else None)
    if pos_idx is None:
        raise ValueError("Cannot determine positive class")
    y_prob = proba[:, pos_idx]
    y_true_bin = (y_val == (positive_label if positive_label is not None else classes[pos_idx])).astype(int)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    pr_auc = average_precision_score(y_true_bin, y_prob)
    brier = brier_score_loss(y_true_bin, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    bal_acc = balanced_accuracy_score(y_true_bin, (y_prob >= best_threshold).astype(int))
    for metric_name, value, thr in [('roc_auc', roc_auc, np.nan), ('pr_auc', pr_auc, np.nan),
                                     ('balanced_accuracy', bal_acc, best_threshold), ('brier_score', brier, np.nan)]:
        results.append({'model': name, 'metric': metric_name, 'value': value, 'threshold': thr})
        if metric_name == primary_metric:
            mlca_runtime.record_metric(primary_metric, value, split='val')

df_metrics = pd.DataFrame(results)
df_metrics['value'] = df_metrics['value'].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
df_metrics['threshold'] = df_metrics['threshold'].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics per candidate model')

best_row = df_metrics[df_metrics['metric'] == primary_metric].sort_values('value', ascending=False).iloc[0]
mlca_runtime.record_fact_candidate('best_model', best_row['model'], source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_row['value'], source='validation_metrics.csv')

print(f"Feature count: {len(feature_cols)}")
print(f"Validation rows: {len(val)}")
print(f"Best model: {best_row['model']}, {primary_metric}={best_row['value']:.4f}")
