import mlca_runtime, json, pickle
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')
task = protocol['task']
primary = task['primary_metric']
if task['type'] != 'binary':
    raise SystemExit('This validation script handles the declared binary task only')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
n = len(train)
for c in list(feature_cols):
    s = train[c]
    if n > 1 and s.nunique(dropna=False) == n and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        feature_cols.remove(c)
X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_specs = [
    ('logistic', load(inp['07_model_train_logistic_0Q0G3DSJ/model.pkl'])),
    ('alternative', load(inp['08_model_train_alternative_EEA21607/alternative_model.pkl'])),
]

m0 = model_specs[0][1]
classes0 = getattr(m0, 'classes_', None)
if classes0 is not None and len(classes0) == 2:
    positive = classes0[1]
else:
    vals = sorted(pd.Series(y_val).dropna().unique())
    positive = vals[1] if len(vals) > 1 else vals[0]

def predict_proba(model, X):
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
        if all(c in X.columns for c in cols):
            return model.predict_proba(X[cols]), getattr(model, 'classes_', None)
    pre = load(inp['05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl'])
    sel = load(inp['06_features_retain_features_SEM4Z3JD/selector_all.pkl'])
    steps = []
    sel_cols = getattr(sel, 'feature_names_in_', None)
    pre_cols = getattr(pre, 'feature_names_in_', None)
    if isinstance(sel, Pipeline) and sel_cols is not None and all(c in X.columns for c in sel_cols):
        steps = [('selector', sel)]
    elif sel_cols is not None and all(c in X.columns for c in sel_cols):
        steps = [('selector', sel)]
    else:
        if pre_cols is not None and all(c in X.columns for c in pre_cols):
            steps.append(('preprocessor', pre))
        steps.append(('selector', sel))
    steps.append(('model', model))
    pipe = Pipeline(steps)
    return pipe.predict_proba(X), getattr(model, 'classes_', None)

rows = []
primary_vals = {}
roc_by_model = {}
for name, model in model_specs:
    proba, classes = predict_proba(model, X_val)
    classes = list(classes) if classes is not None else list(pd.Series(y_val).dropna().unique())
    if positive in classes:
        j = classes.index(positive)
    elif proba.shape[1] == 2:
        j = 1
    else:
        raise ValueError(f'Cannot align positive label for {name}')
    p = proba[:, j]
    yb = (y_val == positive).astype(int).to_numpy()
    roc = roc_auc_score(yb, p)
    pr = average_precision_score(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    if not np.isfinite(t):
        t = float(np.max(p))
    ypred = (p >= t).astype(int)
    bal = balanced_accuracy_score(yb, ypred)
    brier = brier_score_loss(yb, p)
    vals = {'roc_auc': roc, 'pr_auc': pr, 'balanced_accuracy': bal, 'brier_score': brier}
    if primary not in vals:
        raise ValueError(f'Unsupported primary_metric: {primary}')
    primary_vals[name] = vals[primary]
    roc_by_model[name] = roc
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'brier_score']:
        rows.append({
            'model': name,
            'metric': metric,
            'value': float(vals[metric]),
            'threshold': t if metric == 'balanced_accuracy' else np.nan,
        })
    mlca_runtime.record_metric(primary, float(primary_vals[name]), 'val')

higher_better = primary not in {'brier_score', 'log_loss', 'mae', 'rmse'}
best_name = max(primary_vals, key=primary_vals.get) if higher_better else min(primary_vals, key=primary_vals.get)
mlca_runtime.record_metric(primary, float(primary_vals[best_name]), 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(roc_by_model[best_name]), source='validation_metrics.csv')

out = pd.DataFrame(rows, columns=['model', 'metric', 'value', 'threshold'])
for col in ['value', 'threshold']:
    out[col] = out[col].map(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(out, 'validation_metrics.csv', caption='Candidate validation metrics')

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"primary={primary} best_model={best_name} best_val_roc_auc={roc_by_model[best_name]:.4f}")
