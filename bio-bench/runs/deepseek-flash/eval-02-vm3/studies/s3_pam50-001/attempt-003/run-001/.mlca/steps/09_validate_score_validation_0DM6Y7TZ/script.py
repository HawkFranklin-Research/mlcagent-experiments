import mlca_runtime, pickle, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
primary_metric = task['primary_metric']
task_type = task['type']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
def is_key_like(s):
    return s.nunique(dropna=False) == len(s) and not pd.api.types.is_float_dtype(s)
feature_cols = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]

pre = load_pkl(inp['05_preprocess_fit_preprocessing_AP3P14P9/preprocessor.pkl'])
sel = load_pkl(inp['06_features_select_features_NBH1R9WT/selector.pkl'])
candidates = {
    'logistic': load_pkl(inp['07_model_train_logistic_BS7D3474/model.pkl']),
    'alternative': load_pkl(inp['08_model_train_alternative_Y5M9F5DT/alternative_model.pkl']),
}

def contains(obj, tgt):
    if obj is tgt:
        return True
    if isinstance(obj, Pipeline):
        return any(contains(s, tgt) for _, s in obj.steps)
    return False

def make_pipeline(clf):
    if isinstance(clf, Pipeline) and getattr(clf, 'feature_names_in_', None) is not None:
        return clf
    steps = []
    if pre is not None:
        steps.append(('preprocessor', pre))
    if sel is not None and not contains(pre, sel):
        steps.append(('selector', sel))
    steps.append(('clf', clf))
    return Pipeline(steps)

def front_features(pipe):
    objs = [pipe] + ([s for _, s in pipe.steps] if isinstance(pipe, Pipeline) else [])
    for obj in objs:
        fn = getattr(obj, 'feature_names_in_', None)
        if fn is not None:
            return [c for c in fn if c in val.columns]
    return feature_cols

target = targets[0]
y_val = val[target].values
higher_better = primary_metric in ('roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
    'f1', 'f1_macro', 'accuracy', 'balanced_accuracy', 'average_precision', 'pr_auc')

rows = []
metric_vals = {}
for name, clf in candidates.items():
    pipe = make_pipeline(clf)
    cols = front_features(pipe)
    X_val = val[cols]
    proba = pipe.predict_proba(X_val)
    classes = list(pipe.classes_)
    if task_type == 'multiclass':
        present = [c for c in classes if (y_val == c).any()]
        if len(present) < len(classes):
            vals = [roc_auc_score((y_val == c).astype(int), proba[:, classes.index(c)]) for c in present]
            auc = float(np.mean(vals))
            print(f"{name}: missing classes {sorted(set(classes) - set(present))}")
        else:
            auc = float(roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes))
        f1 = float(f1_score(y_val, pipe.predict(X_val), average='macro'))
        rows.append({'model': name, 'metric': 'roc_auc_ovr', 'value': auc, 'threshold': np.nan})
        rows.append({'model': name, 'metric': 'f1_macro', 'value': f1, 'threshold': np.nan})
        metric_vals[name] = auc
    else:
        pos_label = task.get('positive_label', classes[1])
        j = classes.index(pos_label)
        p = proba[:, j]
        yb = (y_val == pos_label).astype(int)
        fpr, tpr, thr = roc_curve(yb, p)
        best_thr = float(thr[int(np.argmax(tpr - fpr))])
        yhat = (p >= best_thr).astype(int)
        rows.append({'model': name, 'metric': 'roc_auc', 'value': float(roc_auc_score(yb, p)), 'threshold': best_thr})
        rows.append({'model': name, 'metric': 'pr_auc', 'value': float(average_precision_score(yb, p)), 'threshold': best_thr})
        rows.append({'model': name, 'metric': 'balanced_accuracy', 'value': float(balanced_accuracy_score(yb, yhat)), 'threshold': best_thr})
        rows.append({'model': name, 'metric': 'brier_score', 'value': float(brier_score_loss(yb, p)), 'threshold': best_thr})
        metric_vals[name] = float(roc_auc_score(yb, p))

best_name = (max if higher_better else min)(metric_vals, key=metric_vals.get)
best_val = metric_vals[best_name]

mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv',
    caption='Validation metrics per candidate model')
mlca_runtime.record_metric(primary_metric, float(best_val), 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_val), source='validation_metrics.csv')
print(f"candidates={list(candidates)} n_features={len(feature_cols)}")
print(f"best={best_name} {primary_metric}={best_val:.4f}")
