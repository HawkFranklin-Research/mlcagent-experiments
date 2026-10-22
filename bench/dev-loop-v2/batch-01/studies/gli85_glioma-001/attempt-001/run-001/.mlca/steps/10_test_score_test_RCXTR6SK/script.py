import mlca_runtime
import pandas as pd, numpy as np, json, pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, confusion_matrix)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
task_type = task.get('type', 'binary')

val_metrics = pd.read_csv(inputs['09_validate_score_validation_V61ZZERD/validation_metrics.csv'])
m07 = json.loads(inputs['07_model_train_logistic_CT3F59V6/manifest.json'].read_text())
m08 = json.loads(inputs['08_model_train_alternative_S4BEA2N2/manifest.json'].read_text())
best_name = ''
try:
    if 'metric' in val_metrics.columns:
        sub = val_metrics[val_metrics['metric'] == primary_metric]
        if len(sub) and 'model' in sub.columns:
            best_name = str(sub.loc[sub['value'].idxmax(), 'model']).lower()
    elif primary_metric in val_metrics.columns and 'model' in val_metrics.columns:
        best_name = str(val_metrics.loc[val_metrics[primary_metric].idxmax(), 'model']).lower()
except Exception:
    pass
use_alt = 'alt' in best_name or '08' in best_name
if use_alt:
    model_path, chosen_manifest = inputs['08_model_train_alternative_S4BEA2N2/alternative_model.pkl'], m08
else:
    model_path, chosen_manifest = inputs['07_model_train_logistic_CT3F59V6/model.pkl'], m07

with open(model_path, 'rb') as f:
    model = pickle.load(f)
test_df = pd.read_parquet(inputs['test'])
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = [c for c in test_df.columns if c not in targets and c not in excluded and c != group_col]
X_test = test_df[feature_cols]
y_true_full = test_df[targets[0]].values
classes = list(model.classes_) if hasattr(model, 'classes_') else list(model.steps[-1][1].classes_)
y_proba = model.predict_proba(X_test)
N = len(y_true_full)
threshold = float(chosen_manifest['threshold']) if chosen_manifest.get('threshold') is not None else 0.5
rng = np.random.default_rng(42)

def boot_ci(fn):
    vals = [v for v in (fn(rng.integers(0, N, N)) for _ in range(1000)) if np.isfinite(v)]
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)

if task_type == 'binary':
    pos_label = task.get('positive_label', classes[1])
    y_bin = (y_true_full == pos_label).astype(int)
    pp = y_proba[:, 1].astype(float)
    y_pred_out = (pp >= threshold).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y_bin, y_pred_out, labels=[0, 1]).ravel()
    value_map = {
        'roc_auc': roc_auc_score(y_bin, pp) if len(np.unique(y_bin)) > 1 else float('nan'),
        'pr_auc': average_precision_score(y_bin, pp),
        'balanced_accuracy': balanced_accuracy_score(y_bin, y_pred_out),
        'brier_score': brier_score_loss(y_bin, pp),
        'sensitivity': tp/(tp+fn_) if (tp+fn_) else float('nan'),
        'specificity': tn/(tn+fp) if (tn+fp) else float('nan'),
    }
    def primary_fn(idx):
        yy, ppb = y_bin[idx], pp[idx]
        yp = (ppb >= threshold).astype(int)
        tn_, fp_, fn2, tp_ = confusion_matrix(yy, yp, labels=[0, 1]).ravel()
        return {
            'roc_auc': lambda: roc_auc_score(yy, ppb) if len(np.unique(yy)) > 1 else np.nan,
            'pr_auc': lambda: average_precision_score(yy, ppb),
            'brier_score': lambda: brier_score_loss(yy, ppb),
            'balanced_accuracy': lambda: balanced_accuracy_score(yy, yp),
            'sensitivity': lambda: tp_/(tp_+fn2) if (tp_+fn2) else np.nan,
            'specificity': lambda: tn_/(tn_+fp_) if (tn_+fp_) else np.nan,
        }.get(primary_metric, lambda: np.nan)()
    ci_low, ci_high = boot_ci(primary_fn)
    primary_value = value_map.get(primary_metric, float('nan'))
    rows = [{'metric': k, 'value': v,
             'ci_low': ci_low if k == primary_metric else np.nan,
             'ci_high': ci_high if k == primary_metric else np.nan} for k, v in value_map.items()]
else:
    y_pred_out = model.predict(X_test)
    def macro_auc(yt, yp):
        vals = [roc_auc_score((yt == c).astype(int), yp[:, i])
                for i, c in enumerate(classes) if c in np.unique(yt) and (yt == c).sum() > 0]
        return float(np.mean(vals)) if vals else np.nan
    roc_auc_val = macro_auc(y_true_full, y_proba)
    f1_val = f1_score(y_true_full, y_pred_out, average='macro')
    is_roc = primary_metric.startswith('roc_auc')
    is_f1 = primary_metric.startswith('f1')
    def primary_fn(idx):
        yt, yp = y_true_full[idx], y_proba[idx]
        if is_roc: return macro_auc(yt, yp)
        if is_f1: return f1_score(yt, np.asarray(classes)[yp.argmax(1)], average='macro')
        return np.nan
    ci_low, ci_high = boot_ci(primary_fn)
    primary_value = roc_auc_val if is_roc else f1_val
    rows = [
        {'metric': 'roc_auc', 'value': roc_auc_val,
         'ci_low': ci_low if is_roc else np.nan, 'ci_high': ci_high if is_roc else np.nan},
        {'metric': 'f1_macro', 'value': f1_val,
         'ci_low': ci_low if is_f1 else np.nan, 'ci_high': ci_high if is_f1 else np.nan},
    ]

metrics_df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')
ci = (ci_low, ci_high) if (np.isfinite(ci_low) and np.isfinite(ci_high)) else None
mlca_runtime.record_metric(primary_metric, float(primary_value), split='test', ci=ci)
roc_row = metrics_df[metrics_df['metric'] == 'roc_auc']
if len(roc_row) and np.isfinite(roc_row.iloc[0]['value']):
    mlca_runtime.record_fact_candidate('test_roc_auc', float(roc_row.iloc[0]['value']), source='test_metrics.csv')

pred_df = pd.DataFrame({'row_id': test_df.index, 'partition': 'test'})
if len(targets) == 1:
    pred_df['y_true'] = y_true_full
    pred_df['y_pred'] = y_pred_out
else:
    for t in targets:
        pred_df[f'y_true__{t}'] = test_df[t].values
        pred_df[f'y_pred__{t}'] = y_pred_out
for i, cls in enumerate(classes):
    pred_df[f'proba__{cls}'] = y_proba[:, i]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"test {primary_metric}={primary_value:.4f} ci=({ci_low:.4f},{ci_high:.4f}) n={N}")
