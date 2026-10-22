import mlca_runtime
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task.get('type', 'binary')

val_metrics = pd.read_csv(inputs['09_validate_score_validation_V61ZZERD/validation_metrics.csv'])
m07 = json.loads(inputs['07_model_train_logistic_CT3F59V6/manifest.json'].read_text())
m08 = json.loads(inputs['08_model_train_alternative_S4BEA2N2/manifest.json'].read_text())
name07 = str(m07.get('model_name', m07.get('name', 'logistic')))
name08 = str(m08.get('model_name', m08.get('name', 'alternative')))

if primary_metric in val_metrics.columns:
    idx = val_metrics[primary_metric].idxmax()
    best_row = val_metrics.loc[idx]
    best_model = str(best_row.get('model', best_row.get('model_name', best_row.get('name', name07))))
elif 'metric' in val_metrics.columns:
    sub = val_metrics[val_metrics['metric'] == primary_metric]
    if len(sub) == 0:
        raise ValueError(f"No rows for metric={primary_metric}; present={val_metrics['metric'].unique().tolist()}")
    best_row = sub.loc[sub['value'].idxmax()]
    best_model = str(best_row['model']) if 'model' in sub.columns else name07
else:
    raise ValueError(f"Cannot parse validation_metrics.csv columns={val_metrics.columns.tolist()}")

if str(best_model) == name08 or ('logistic' not in str(best_model).lower() and str(best_model) != name07):
    model_path = inputs['08_model_train_alternative_S4BEA2N2/alternative_model.pkl']
    chosen_manifest = m08
else:
    model_path = inputs['07_model_train_logistic_CT3F59V6/model.pkl']
    chosen_manifest = m07

with open(model_path, 'rb') as f:
    model = pickle.load(f)

if isinstance(model, Pipeline):
    steps = model.named_steps
    if any(k in steps for k in ['preprocessor', 'preprocess', 'preprocessing']) and \
       any(k in steps for k in ['selector', 'select', 'feature_selector']):
        pipeline = model
    else:
        with open(inputs['05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl'], 'rb') as f:
            pre = pickle.load(f)
        with open(inputs['06_features_select_features_K8ATBX97/selector.pkl'], 'rb') as f:
            sel = pickle.load(f)
        m_steps = model.steps if isinstance(model, Pipeline) else [('model', model)]
        pipeline = Pipeline([('preprocessor', pre), ('selector', sel)] + list(m_steps))
else:
    with open(inputs['05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inputs['06_features_select_features_K8ATBX97/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    pipeline = Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)])

test_df = pd.read_parquet(inputs['test'])
if hasattr(pipeline.steps[0][1], 'feature_names_in_'):
    feature_cols = list(pipeline.steps[0][1].feature_names_in_)
else:
    feature_cols = [c for c in test_df.columns if c not in targets and c not in excluded
                    and (group_col is None or c != group_col)]

X_test = test_df[feature_cols].copy()
y_true_full = test_df[targets[0]].values

threshold = 0.5
if 'threshold' in chosen_manifest:
    threshold = float(chosen_manifest['threshold'])
elif 'threshold' in best_row.index and pd.notna(best_row.get('threshold')):
    threshold = float(best_row['threshold'])

classes = pipeline.steps[-1][1].classes_
y_proba = pipeline.predict_proba(X_test)
rng = np.random.default_rng(42)
n = len(y_true_full)
boot = []

if task_type == 'binary':
    pos_label = classes[1]
    y_bin = (y_true_full == pos_label).astype(int)
    pp = y_proba[:, 1]
    y_pred_bin = (pp >= threshold).astype(int)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_bin[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_bin[idx], pp[idx]))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    roc_auc = roc_auc_score(y_bin, pp)
    pr_auc = average_precision_score(y_bin, pp)
    brier = brier_score_loss(y_bin, pp)
    bal_acc = balanced_accuracy_score(y_bin, y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    metrics_df = pd.DataFrame([
        {'metric': 'roc_auc', 'value': roc_auc, 'ci_low': ci_low, 'ci_high': ci_high},
        {'metric': 'pr_auc', 'value': pr_auc, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'balanced_accuracy', 'value': bal_acc, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'brier_score', 'value': brier, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'sensitivity', 'value': sens, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'specificity', 'value': spec, 'ci_low': np.nan, 'ci_high': np.nan},
    ])
    y_pred_out = y_pred_bin
    primary_value = roc_auc if primary_metric == 'roc_auc' else bal_acc
else:
    y_pred_out = pipeline.predict(X_test)
    present = np.unique(y_true_full)
    def macro_auc(y_t, y_p):
        pres_idx = [i for i, c in enumerate(classes) if c in np.unique(y_t)]
        if len(pres_idx) < len(classes):
            vals = []
            for i in pres_idx:
                yb = (y_t == classes[i]).astype(int)
                if len(np.unique(yb)) < 2:
                    continue
                vals.append(roc_auc_score(yb, y_p[:, i]))
            return float(np.mean(vals)) if vals else np.nan
        return roc_auc_score(y_t, y_p, multi_class='ovr', average='macro', labels=classes)
    macro_auc_val = macro_auc(y_true_full, y_proba)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        boot.append(macro_auc(y_true_full[idx], y_proba[idx]))
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
    macro_f1 = f1_score(y_true_full, y_pred_out, average='macro')
    metrics_df = pd.DataFrame([
        {'metric': 'roc_auc', 'value': macro_auc_val, 'ci_low': ci_low, 'ci_high': ci_high},
        {'metric': 'f1_macro', 'value': macro_f1, 'ci_low': np.nan, 'ci_high': np.nan},
    ])
    primary_value = macro_auc_val if primary_metric == 'roc_auc' else macro_f1

mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')
mlca_runtime.record_metric(primary_metric, float(primary_value), split='test', ci=(float(ci_low), float(ci_high)))
mlca_runtime.record_fact_candidate('test_roc_auc', float(metrics_df.loc[0, 'value']), source='test_metrics.csv')

pred_df = pd.DataFrame({'row_id': test_df.index, 'partition': 'test',
                        'y_true': y_true_full, 'y_pred': y_pred_out})
for i, cls in enumerate(classes):
    pred_df[f'proba__{cls}'] = y_proba[:, i]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"Test {primary_metric}: {primary_value:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Chosen model: {best_model}; n_test={n}")
