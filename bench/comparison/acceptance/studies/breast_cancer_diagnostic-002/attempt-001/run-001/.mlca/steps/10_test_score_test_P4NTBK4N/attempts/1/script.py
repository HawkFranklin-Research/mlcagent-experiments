import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, recall_score, roc_curve)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
task = protocol['task']
primary = task['primary_metric']
n_classes = task.get('n_classes', 2)

test = pd.read_parquet(inputs['test'])
val = pd.read_parquet(inputs['val'])

exclude_set = set(targets) | set(excluded)
if group: exclude_set.add(group)
feature_cols = [c for c in test.columns if c not in exclude_set]
for c in list(feature_cols):
    if test[c].dtype == object and test[c].nunique() == len(test):
        feature_cols.remove(c)
    elif test[c].dtype.kind in 'iu' and test[c].nunique() == len(test) and any(k in c.lower() for k in ['id','key','code']):
        feature_cols.remove(c)

model_path = inputs['latest_model']
try:
    mf = json.loads(inputs['09_validate_score_validation_2AQ6E156/manifest.json'].read_text())
    for k in ['selected_model','best_model','winner']:
        if k in mf:
            sel = str(mf[k]).lower()
            if 'alternative' in sel or 'latest' in sel:
                model_path = inputs['08_model_train_alternative_BY729WV6/alternative_model.pkl']
            elif 'logistic' in sel:
                model_path = inputs['07_model_train_logistic_NCC1WZ5Q/model.pkl']
            break
except Exception:
    pass

if model_path == inputs['latest_model']:
    try:
        vm = pd.read_csv(inputs['09_validate_score_validation_2AQ6E156/validation_metrics.csv'])
        if 'model' in vm.columns and primary in vm.columns:
            higher = primary in ['roc_auc','pr_auc','balanced_accuracy','f1','accuracy']
            best = vm.loc[vm[primary].idxmax() if higher else vm[primary].idxmin()]
            name = str(best['model']).lower()
            if 'logistic' in name:
                model_path = inputs['07_model_train_logistic_NCC1WZ5Q/model.pkl']
            elif 'alternative' in name or 'latest' in name:
                model_path = inputs['08_model_train_alternative_BY729WV6/alternative_model.pkl']
    except Exception:
        pass

model = pickle.load(open(model_path, 'rb'))
pos_label = task.get('positive_label') or task.get('positive_class')
if pos_label is None:
    pos_label = model.classes_[1]
pos_idx = list(model.classes_).index(pos_label)

threshold = None
try:
    mf = json.loads(inputs['09_validate_score_validation_2AQ6E156/manifest.json'].read_text())
    for k in ['threshold','best_threshold','frozen_threshold','optimal_threshold']:
        if k in mf:
            threshold = float(mf[k])
            break
except Exception:
    pass
if threshold is None:
    X_val = val[feature_cols]
    y_val = val[targets[0]]
    y_val_bin = (y_val == pos_label).astype(int)
    val_proba = model.predict_proba(X_val)[:, pos_idx]
    fpr, tpr, thr = roc_curve(y_val_bin, val_proba)
    threshold = float(thr[np.argmax(tpr - fpr)])

X_test = test[feature_cols]
y_test = test[targets[0]]
y_test_bin = (y_test == pos_label).astype(int)
test_proba = model.predict_proba(X_test)[:, pos_idx]
test_pred = (test_proba >= threshold).astype(int)

def metrics(y_true, y_prob, y_pred):
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_prob),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

full = metrics(y_test_bin, test_proba, test_pred)
rng = np.random.default_rng(42)
boot = {k: [] for k in full}
n = len(test)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb = y_test_bin.values[idx]
    pb = test_proba[idx]
    db = test_pred[idx]
    if len(np.unique(yb)) < 2:
        continue
    m = metrics(yb, pb, db)
    for k in boot:
        boot[k].append(m[k])

rows = []
for k in full:
    v = boot[k]
    rows.append({'metric': k, 'value': full[k],
                 'ci_low': np.percentile(v, 2.5) if v else np.nan,
                 'ci_high': np.percentile(v, 97.5) if v else np.nan})
df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

prim_row = df[df['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, prim_row['value'], 'test',
                           ci=(prim_row['ci_low'], prim_row['ci_high']))
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', prim_row['value'], source='test_metrics.csv')

print(f"Test {primary}: {prim_row['value']:.4f} ({prim_row['ci_low']:.4f}-{prim_row['ci_high']:.4f})")
