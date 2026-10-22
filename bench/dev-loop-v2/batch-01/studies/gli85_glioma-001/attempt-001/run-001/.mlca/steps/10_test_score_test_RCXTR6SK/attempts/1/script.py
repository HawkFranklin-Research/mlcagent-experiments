import mlca_runtime
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

# Load protocol
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task['type']
n_classes = task.get('n_classes', None)

# Read validation metrics to select best model
val_metrics = pd.read_csv(inputs['09_validate_score_validation_V61ZZERD/validation_metrics.csv'])
if primary_metric not in val_metrics.columns:
    raise ValueError(f"Primary metric {primary_metric} not in validation metrics columns: {val_metrics.columns}")
model_col = None
for cand in ['model', 'model_name', 'name']:
    if cand in val_metrics.columns:
        model_col = cand
        break
if model_col:
    best_row = val_metrics.loc[val_metrics[primary_metric].idxmax()]
    best_model_name = best_row[model_col]
else:
    best_model_name = val_metrics[primary_metric].idxmax()
print(f"Best model on validation {primary_metric}: {best_model_name}")

# Map best model to path
if 'logistic' in str(best_model_name).lower():
    model_path = inputs['07_model_train_logistic_CT3F59V6/model.pkl']
    manifest_path = inputs['07_model_train_logistic_CT3F59V6/manifest.json']
elif 'alternative' in str(best_model_name).lower() or 'alt' in str(best_model_name).lower():
    model_path = inputs['08_model_train_alternative_S4BEA2N2/alternative_model.pkl']
    manifest_path = inputs['08_model_train_alternative_S4BEA2N2/manifest.json']
else:
    model_path = inputs['latest_model']
    manifest_path = inputs['08_model_train_alternative_S4BEA2N2/manifest.json']

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Build final pipeline
def build_pipeline(model):
    if isinstance(model, Pipeline):
        steps = model.named_steps
        has_preproc = any(k in steps for k in ['preprocessor', 'preprocess', 'preprocessing'])
        has_selector = any(k in steps for k in ['selector', 'select', 'feature_selector'])
        if has_preproc and has_selector:
            return model
        elif has_preproc and not has_selector:
            with open(inputs['06_features_select_features_K8ATBX97/selector.pkl'], 'rb') as f:
                selector = pickle.load(f)
            new_steps = list(model.steps[:-1]) + [('selector', selector)] + [model.steps[-1]]
            return Pipeline(new_steps)
        elif has_selector and not has_preproc:
            with open(inputs['05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl'], 'rb') as f:
                preprocessor = pickle.load(f)
            new_steps = [('preprocessor', preprocessor)] + list(model.steps)
            return Pipeline(new_steps)
        else:
            with open(inputs['05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl'], 'rb') as f:
                preprocessor = pickle.load(f)
            with open(inputs['06_features_select_features_K8ATBX97/selector.pkl'], 'rb') as f:
                selector = pickle.load(f)
            new_steps = [('preprocessor', preprocessor), ('selector', selector)] + list(model.steps)
            return Pipeline(new_steps)
    else:
        with open(inputs['05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl'], 'rb') as f:
            preprocessor = pickle.load(f)
        with open(inputs['06_features_select_features_K8ATBX97/selector.pkl'], 'rb') as f:
            selector = pickle.load(f)
        return Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])

pipeline = build_pipeline(model)

# Determine feature columns
test_df = pd.read_parquet(inputs['test'])
all_cols = test_df.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and (group_col is None or c != group_col)]
for c in feature_cols[:]:
    s = test_df[c]
    if (s.dtype == 'object' or s.dtype.name in ('string', 'category')) and s.nunique() == len(s):
        feature_cols.remove(c)
    elif s.dtype.kind in 'iu' and s.nunique() == len(s):
        feature_cols.remove(c)

# If pipeline has feature_names_in_, use those columns
if hasattr(pipeline, 'feature_names_in_'):
    feature_cols = pipeline.feature_names_in_.tolist()
elif hasattr(pipeline.steps[0][1], 'feature_names_in_'):
    feature_cols = pipeline.steps[0][1].feature_names_in_.tolist()

X_test = test_df[feature_cols].copy()
y_true_full = test_df[targets[0]].values

# Get frozen threshold from validation metrics or manifest
threshold = 0.5
if 'threshold' in val_metrics.columns and model_col:
    threshold = best_row['threshold']
else:
    try:
        manifest = json.loads(manifest_path.read_text())
        threshold = manifest.get('threshold', 0.5)
    except Exception:
        pass

# Final estimator for classes
final_est = pipeline.steps[-1][1] if isinstance(pipeline, Pipeline) else pipeline
classes = final_est.classes_

if task_type == 'binary':
    y_proba = pipeline.predict_proba(X_test)
    pos_label = classes[1]
    y_true_bin = (y_true_full == pos_label).astype(int)
    proba_pos = y_proba[:, 1]
    roc_auc = roc_auc_score(y_true_bin, proba_pos)
    pr_auc = average_precision_score(y_true_bin, proba_pos)
    brier = brier_score_loss(y_true_bin, proba_pos)
    y_pred = (proba_pos >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    rng = np.random.default_rng(42)
    n = len(y_true_bin)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true_bin[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true_bin[idx], proba_pos[idx]))
    ci_low = np.percentile(boot_aucs, 2.5)
    ci_high = np.percentile(boot_aucs, 97.5)

    metrics_df = pd.DataFrame([
        {'metric': 'roc_auc', 'value': roc_auc, 'ci_low': ci_low, 'ci_high': ci_high},
        {'metric': 'pr_auc', 'value': pr_auc, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'balanced_accuracy', 'value': bal_acc, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'brier_score', 'value': brier, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'sensitivity', 'value': sensitivity, 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'specificity', 'value': specificity, 'ci_low': np.nan, 'ci_high': np.nan},
    ])
    mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI for ROC AUC')
    mlca_runtime.record_metric(primary_metric, roc_auc, split='test', ci=(ci_low, ci_high))
    mlca_runtime.record_fact_candidate('test_roc_auc', roc_auc, source='test_metrics.csv')

    pred_df = pd.DataFrame({'row_id': test_df.index, 'partition': 'test',
                            'y_true': y_true_full, 'y_pred': y_pred})
    for i, cls in enumerate(classes):
        pred_df[f'proba__{cls}'] = y_proba[:, i]
    mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

elif task_type == 'multiclass':
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)
    present_classes = np.unique(y_true_full)
    present_idx = [i for i, c in enumerate(classes) if c in present_classes]
    if len(present_idx) < len(classes):
        aucs = []
        for i in present_idx:
            c = classes[i]
            y_bin = (y_true_full == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            aucs.append(roc_auc_score(y_bin, y_proba[:, i]))
        macro_auc = np.mean(aucs) if aucs else np.nan
        print(f"Classes left out: {set(classes) - set(present_classes)}")
    else:
        macro_auc = roc_auc_score(y_true_full, y_proba, multi_class='ovr', average='macro', labels=classes)
    macro_f1 = f1_score(y_true_full, y_pred, average='macro')

    rng = np.random.default_rng(42)
    n = len(y_true_full)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        y_true_b = y_true_full[idx]
        y_proba_b = y_proba[idx]
        present_classes_b = np.unique(y_true_b)
        present_idx_b = [i for i, c in enumerate(classes) if c in present_classes_b]
        if len(present_idx_b) < len(classes):
            aucs_b = []
            for i in present_idx_b:
                c = classes[i]
                y_bin = (y_true_b == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                aucs_b.append(roc_auc_score(y_bin, y_proba_b[:, i]))
            boot_auc = np.mean(aucs_b) if aucs_b else np.nan
        else:
            boot_auc = roc_auc_score(y_true_b, y_proba_b, multi_class='ovr', average='macro', labels=classes)
        boot_aucs.append(boot_auc)
    ci_low = np.nanpercentile(boot_aucs, 2.5)
    ci_high = np.nanpercentile(boot_aucs, 97.5)

    metrics_df = pd.DataFrame([
        {'metric': 'roc_auc', 'value': macro_auc, 'ci_low': ci_low, 'ci_high': ci_high},
        {'metric': 'f1_macro', 'value': macro_f1, 'ci_low': np.nan, 'ci_high': np.nan},
    ])
    mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI for macro ROC AUC')
    mlca_runtime.record_metric(primary_metric, macro_auc, split='test', ci=(ci_low, ci_high))
    mlca_runtime.record_fact_candidate('test_roc_auc', macro_auc, source='test_metrics.csv')

    pred_df = pd.DataFrame({'row_id': test_df.index, 'partition': 'test',
                            'y_true': y_true_full, 'y_pred': y_pred})
    for i, cls in enumerate(classes):
        pred_df[f'proba__{cls}'] = y_proba[:, i]
    mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"Test {primary_metric}: {metrics_df.loc[0, 'value']:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
