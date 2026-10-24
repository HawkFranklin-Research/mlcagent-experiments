import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
task = protocol['task']
primary = task['primary_metric']
targets = list(protocol['targets'])
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')
train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])
y_col = targets[0]

feature_cols = [c for c in train.columns if c not in set(targets) | excluded and c != group]
feature_cols = [c for c in feature_cols if not (train[c].dtype == object and train[c].nunique(dropna=True) == len(train))]

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(paths['05_preprocess_fit_preprocessing_WM2D925J/preprocessor.pkl'])
sel = load_pkl(paths['06_features_select_features_8X366KXZ/selector.pkl'])
candidates = [
    ('logistic', load_pkl(paths['07_model_train_logistic_RTDQBRJK/model.pkl'])),
    ('alternative', load_pkl(paths['08_model_train_alternative_8FW5JDB6/alternative_model.pkl'])),
]

def first_step(obj):
    return list(obj.named_steps.values())[0] if isinstance(obj, Pipeline) else obj

def final_estimator(obj):
    while isinstance(obj, Pipeline):
        obj = list(obj.named_steps.values())[-1]
    return obj

def expected_features(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    f = first_step(obj)
    return list(f.feature_names_in_) if hasattr(f, 'feature_names_in_') else None

def predict_candidate(model):
    options = [model]
    if sel is not None:
        options.append(Pipeline([('selector', sel), ('model', model)]))
    if pre is not None and sel is not None:
        options.append(Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)]))
    if pre is not None:
        options.append(Pipeline([('preprocessor', pre), ('model', model)]))
    errors = []
    for obj in options:
        cols = expected_features(obj)
        if cols is not None:
            if not set(cols).issubset(val.columns):
                continue
            if set(cols) & (set(targets) | excluded | ({group} if group else set())):
                continue
            X = val[cols]
        else:
            X = val[feature_cols]
        try:
            return obj, obj.predict_proba(X)
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError('No candidate pipeline could predict: ' + ' | '.join(errors[:3]))

rows = []
for name, model in candidates:
    pipe, proba = predict_candidate(model)
    classes = getattr(final_estimator(pipe), 'classes_', None)
    if classes is None:
        classes = getattr(pipe, 'classes_', None)
    if classes is None or len(classes) != 2:
        raise ValueError(f'{name}: expected fitted binary classes_')
    pos = classes[1]
    j = list(classes).index(pos)
    y = val[y_col].to_numpy()
    yb = (y == pos).astype(int)
    p = proba[:, j]
    auc = roc_auc_score(yb, p)
    ap = average_precision_score(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    jstat = tpr - fpr
    finite = np.isfinite(thr)
    k = int(np.argmax(np.where(finite, jstat, -np.inf)))
    youden = float(thr[k]) if finite[k] else np.nan
    pred = p >= youden
    ba = balanced_accuracy_score(yb, pred)
    brier = brier_score_loss(yb, p)
    rows.extend([
        {'model': name, 'metric': primary, 'value': float(auc), 'threshold': np.nan},
        {'model': name, 'metric': 'pr_auc', 'value': float(ap), 'threshold': np.nan},
        {'model': name, 'metric': 'balanced_accuracy', 'value': float(ba), 'threshold': youden},
        {'model': name, 'metric': 'brier_score', 'value': float(brier), 'threshold': np.nan},
    ])

df = pd.DataFrame(rows)
for col in ('value', 'threshold'):
    df[col] = pd.to_numeric(df[col], errors='coerce').round(4)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics by candidate')

primary_rows = df[df['metric'] == primary]
higher_better = primary not in {'brier_score', 'log_loss'}
best_idx = primary_rows['value'].idxmax() if higher_better else primary_rows['value'].idxmin()
best_row = primary_rows.loc[best_idx]
best_name = str(best_row['model'])
best_value = float(best_row['value'])
mlca_runtime.record_metric(primary, best_value, split='val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('best_val_roc_auc', best_value, source='validation_metrics.csv')
print(f'val={val.shape}')
print(f'best_model={best_name}')
print(f'best_val_{primary}={best_value:.4g}')
