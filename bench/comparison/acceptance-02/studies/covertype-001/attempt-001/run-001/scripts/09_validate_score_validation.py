import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
with open(inp['_protocol']) as f:
    proto = json.load(f)

for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise KeyError(f"Missing {k}; available top-level keys: {list(proto)}")
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')
task = proto['task']
if 'primary_metric' not in task:
    raise KeyError(f"Missing task.primary_metric; available top-level keys: {list(proto)}")
primary = task['primary_metric']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
target = targets[0]
if target not in train or target not in val:
    raise KeyError(f"Target {target!r} not in partitions")

def key_like(s):
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    return False

skip = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in skip and not key_like(train[c])]
Xv = val[features]
y = val[target].to_numpy()

def pkl(k):
    with open(inp[k], 'rb') as f:
        return pickle.load(f)

pre = pkl('05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl')
sel = pkl('06_features_select_features_53HVANE8/selector.pkl')
models = {
    'logistic': pkl('07_model_train_logistic_Z4Z606HV/model.pkl'),
    'alternative': pkl('08_model_train_alternative_JKAAJDG0/alternative_model.pkl'),
}

def get_classes(m):
    if hasattr(m, 'classes_'):
        return np.asarray(m.classes_)
    if isinstance(m, Pipeline):
        return get_classes(m.steps[-1][1])
    raise AttributeError('model has no classes_')

def transform(X):
    sel_fn = getattr(sel, 'feature_names_in_', None)
    if sel_fn is not None and list(sel_fn) == list(features):
        return sel.transform(X)
    Xt = X
    if pre is not None:
        Xt = pre.transform(X)
        if sel is not None and hasattr(sel, 'feature_names_in_') and not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt, columns=sel.feature_names_in_, index=X.index)
    if sel is not None:
        Xt = sel.transform(Xt)
    return Xt

def predict(m, X):
    fn = getattr(m, 'feature_names_in_', None)
    if fn is not None and list(fn) != list(features):
        X2 = transform(X)
        return m.predict_proba(X2), m.predict(X2)
    try:
        return m.predict_proba(X), m.predict(X)
    except ValueError:
        X2 = transform(X)
        return m.predict_proba(X2), m.predict(X2)

def macro_ovr_auc(y_true, proba, classes):
    classes = np.asarray(classes)
    present = [c for c in classes if np.any(y_true == c)]
    if len(present) == len(classes):
        return float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro', labels=classes))
    vals = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        vals.append(roc_auc_score((y_true == c).astype(int), proba[:, j]))
    return float(np.mean(vals))

def metric_val(name, y_true, pred, proba, classes):
    if name in ('roc_auc_ovr', 'roc_auc', 'roc_auc_ovo'):
        return macro_ovr_auc(y_true, proba, classes)
    if name in ('f1_macro', 'f1'):
        return float(f1_score(y_true, pred, labels=classes, average='macro', zero_division=0))
    if name == 'accuracy':
        return float(accuracy_score(y_true, pred))
    if name == 'balanced_accuracy':
        return float(balanced_accuracy_score(y_true, pred))
    raise ValueError(f'Unsupported primary metric: {name}')

rows = []
primary_vals = {}
for name, m in models.items():
    proba, pred = predict(m, Xv)
    classes = get_classes(m)
    f1 = float(f1_score(y, pred, labels=classes, average='macro', zero_division=0))
    pv = metric_val(primary, y, pred, proba, classes)
    primary_vals[name] = pv
    rows.append({'model': name, 'metric': primary, 'value': pv, 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'f1_macro', 'value': f1, 'threshold': np.nan})
    mlca_runtime.record_metric(primary, float(pv), 'val')

df = pd.DataFrame(rows, columns=['model', 'metric', 'value', 'threshold'])
df['value'] = df['value'].map(lambda v: float(f'{v:.4g}'))
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics by model')

lower = any(k in primary.lower() for k in ('brier', 'log_loss', 'loss', 'error', 'mae', 'mse', 'rmse'))
best = min(primary_vals, key=primary_vals.get) if lower else max(primary_vals, key=primary_vals.get)
mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
if primary in ('roc_auc', 'roc_auc_ovr', 'roc_auc_ovo'):
    mlca_runtime.record_fact_candidate('best_val_roc_auc', float(primary_vals[best]), source='validation_metrics.csv')

print(f"train={train.shape} val={val.shape}")
print(f"features={len(features)} best={best} {primary}={primary_vals[best]:.4g}")
