import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target_cols = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
n_classes = task.get('n_classes', 2)
task_type = task['type']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

feature_cols = [c for c in train.columns
                if c not in target_cols and c not in excluded and c != group_col]

with open(inputs['05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_6Y9H969T/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_logistic_BMNCC8GS/model.pkl'], 'rb') as f:
    model = pickle.load(f)
with open(inputs['08_model_train_alternative_KFV9SB0C/alternative_model.pkl'], 'rb') as f:
    alt_model = pickle.load(f)

def build_pipeline(pre, sel, est):
    steps = []
    if isinstance(est, Pipeline) and any(s is sel for s in est.named_steps.values()):
        if any(s is pre for s in est.named_steps.values()):
            return est
        steps.extend(pre.steps) if isinstance(pre, Pipeline) else steps.append(('pre', pre))
        steps.extend(est.steps)
        return Pipeline(steps)
    if isinstance(sel, Pipeline) and any(s is pre for s in sel.named_steps.values()):
        if isinstance(est, Pipeline):
            steps.extend(sel.steps); steps.extend(est.steps); return Pipeline(steps)
        return Pipeline([('sel', sel), ('model', est)])
    steps.extend(pre.steps) if isinstance(pre, Pipeline) else steps.append(('pre', pre))
    steps.extend(sel.steps) if isinstance(sel, Pipeline) else steps.append(('sel', sel))
    steps.extend(est.steps) if isinstance(est, Pipeline) else steps.append(('model', est))
    return Pipeline(steps)

pipe_main = build_pipeline(preprocessor, selector, model)
pipe_alt = build_pipeline(preprocessor, selector, alt_model)

def get_scores(pipe, X):
    proba = pipe.predict_proba(X[feature_cols])
    classes = pipe.steps[-1][1].classes_
    return proba, classes

def get_pos_idx(classes, positive_label):
    if positive_label is None:
        return 1
    return np.where(classes == positive_label)[0][0]

positive_label = protocol.get('positive_label') or task.get('positive_label')
y_val = val[target_cols[0]].values
proba_main, classes_main = get_scores(pipe_main, val)
proba_alt, classes_alt = get_scores(pipe_alt, val)
if positive_label is None:
    positive_label = classes_main[1]
y_val_bin = (y_val == positive_label).astype(int)

if task_type == 'binary':
    val_metric_main = roc_auc_score(y_val_bin, proba_main[:, get_pos_idx(classes_main, positive_label)])
    val_metric_alt = roc_auc_score(y_val_bin, proba_alt[:, get_pos_idx(classes_alt, positive_label)])
else:
    val_metric_main = roc_auc_score(y_val, proba_main, multi_class='ovr', average='macro', labels=classes_main)
    val_metric_alt = roc_auc_score(y_val, proba_alt, multi_class='ovr', average='macro', labels=classes_alt)

selected_pipe = pipe_alt if val_metric_alt > val_metric_main else pipe_main
selected_name = 'alternative' if val_metric_alt > val_metric_main else 'logistic'

threshold = None
try:
    vm = pd.read_csv(inputs['09_validate_score_validation_0H12DC4P/validation_metrics.csv'])
    if 'threshold' in vm.columns:
        threshold = float(vm['threshold'].iloc[0])
except Exception:
    pass
if threshold is None:
    try:
        man = json.loads(inputs['09_validate_score_validation_0H12DC4P/manifest.json'].read_text())
        def find_th(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == 'threshold' and isinstance(v, (int, float)):
                        return v
                    r = find_th(v)
                    if r is not None: return r
            elif isinstance(d, list):
                for it in d:
                    r = find_th(it)
                    if r is not None: return r
            return None
        threshold = find_th(man)
    except Exception:
        pass
if threshold is None and task_type == 'binary':
    yv_proba = selected_pipe.predict_proba(val[feature_cols])[:, get_pos_idx(selected_pipe.steps[-1][1].classes_, positive_label)]
    best_t, best_ba = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        ba = balanced_accuracy_score(y_val_bin, (yv_proba >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    threshold = best_t

X_test = test[feature_cols]
y_test = test[target_cols[0]].values
proba_test, classes_test = get_scores(selected_pipe, test)
if task_type == 'binary':
    y_test_bin = (y_test == positive_label).astype(int)
    scores_test = proba_test[:, get_pos_idx(classes_test, positive_label)]
    test_metric = roc_auc_score(y_test_bin, scores_test)
else:
    scores_test = proba_test
    test_metric = roc_auc_score(y_test, scores_test, multi_class='ovr', average='macro', labels=classes_test)

rng = np.random.RandomState(42)
n_boot = 1000
metrics_dict = {}
if task_type == 'binary':
    def compute_binary(y_t, y_s, t):
        roc = roc_auc_score(y_t, y_s)
        pr = average_precision_score(y_t, y_s)
        brier = brier_score_loss(y_t, y_s)
        y_pred = (y_s >= t).astype(int)
        ba = balanced_accuracy_score(y_t, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return {'roc_auc': roc, 'pr_auc': pr, 'brier': brier,
                'balanced_accuracy': ba, 'sensitivity': sens, 'specificity': spec}
    metrics_dict['point'] = compute_binary(y_test_bin, scores_test, threshold)
    boot = {k: [] for k in metrics_dict['point']}
    n = len(y_test_bin)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_t, y_s = y_test_bin[idx], scores_test[idx]
        if len(np.unique(y_t)) < 2: continue
        m = compute_binary(y_t, y_s, threshold)
        for k, v in m.items():
            if not np.isnan(v): boot[k].append(v)
    for k in boot:
        metrics_dict[f'{k}_ci_low'] = np.percentile(boot[k], 2.5) if len(boot[k]) > 1 else np.nan
        metrics_dict[f'{k}_ci_high'] = np.percentile(boot[k], 97.5) if len(boot[k]) > 1 else np.nan
else:
    def compute_multi(y_t, y_s):
        roc = roc_auc_score(y_t, y_s, multi_class='ovr', average='macro', labels=classes_test)
        y_pred = classes_test[np.argmax(y_s, axis=1)]
        f1 = f1_score(y_t, y_pred, average='macro')
        return {'roc_auc_ovr': roc, 'macro_f1': f1}
    metrics_dict['point'] = compute_multi(y_test, scores_test)
    boot = {k: [] for k in metrics_dict['point']}
    n = len(y_test)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_t, y_s = y_test[idx], scores_test[idx]
        if len(np.unique(y_t)) < len(classes_test): continue
        m = compute_multi(y_t, y_s)
        for k, v in m.items(): boot[k].append(v)
    for k in boot:
        metrics_dict[f'{k}_ci_low'] = np.percentile(boot[k], 2.5)
        metrics_dict[f'{k}_ci_high'] = np.percentile(boot[k], 97.5)

rows = []
for metric in metrics_dict['point']:
    rows.append({
        'metric': metric,
        'value': metrics_dict['point'][metric],
        'ci_low': metrics_dict.get(f'{metric}_ci_low', np.nan),
        'ci_high': metrics_dict.get(f'{metric}_ci_high', np.nan)
    })
test_metrics_df = pd.DataFrame(rows)
for col in ['value', 'ci_low', 'ci_high']:
    test_metrics_df[col] = test_metrics_df[col].apply(lambda x: float(f'{x:.4g}') if pd.notnull(x) else x)
mlca_runtime.save_table(test_metrics_df, 'test_metrics.csv', caption='Test metrics with bootstrap CI')

prim_value = metrics_dict['point'].get(primary_metric, test_metric)
prim_low = metrics_dict.get(f'{primary_metric}_ci_low', np.nan)
prim_high = metrics_dict.get(f'{primary_metric}_ci_high', np.nan)
mlca_runtime.record_metric(primary_metric, prim_value, split='test', ci=[prim_low, prim_high])
if primary_metric == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', prim_value, source='test_metrics.csv')

print(f"Selected: {selected_name}, features: {len(feature_cols)}")
print(f"Positive label: {positive_label}, threshold: {threshold:.4f}")
print(f"Test {primary_metric}: {prim_value:.4f} [{prim_low:.4f}, {prim_high:.4f}]")
