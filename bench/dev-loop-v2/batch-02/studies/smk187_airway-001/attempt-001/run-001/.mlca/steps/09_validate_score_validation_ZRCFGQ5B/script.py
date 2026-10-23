import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs['_protocol']).read_text())
task = protocol['task']; targets = list(protocol['targets']); target = targets[0]
excluded = list(protocol.get('excluded_columns', [])); group_col = protocol.get('group_column')
primary = task['primary_metric']; task_type = task.get('type', 'binary')
train = pd.read_parquet(inputs['train']); val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col: drop.add(group_col)
def keylike(s):
    if s.isna().any(): return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ('model', 'pipeline', 'estimator', 'best_estimator'):
            if k in obj: return unwrap(obj[k])
    return obj

def load(path):
    return unwrap(pickle.loads(Path(path).read_bytes()))

pre = load(inputs['05_preprocess_fit_preprocessing_4ZKXP498/preprocessor.pkl'])
sel = load(inputs['06_features_select_features_E9CTG9EJ/selector.pkl'])
models = {
    'logistic': load(inputs['07_model_train_logistic_KVA6214M/model.pkl']),
    'alternative': load(inputs['08_model_train_alternative_DY6PAN53/alternative_model.pkl']),
}

def get_classes(model):
    c = getattr(model, 'classes_', None)
    if c is not None: return list(c)
    if isinstance(model, Pipeline):
        c = getattr(model.steps[-1][1], 'classes_', None)
        if c is not None: return list(c)
    return None

def as_frame(Z, obj, index, fallback=None):
    if isinstance(Z, pd.DataFrame): return Z
    Z = np.asarray(Z)
    cols = None
    if hasattr(obj, 'get_feature_names_out'):
        try: cols = list(obj.get_feature_names_out())
        except Exception: pass
    if cols is None and fallback is not None and len(fallback) == Z.shape[1]:
        cols = list(fallback)
    if cols is None: cols = [f'f{i}' for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols, index=index)

def score_model(model):
    model = unwrap(model)
    names = getattr(model, 'feature_names_in_', None)
    if names is not None and all(n in val.columns for n in names):
        try: return model.predict_proba(val[list(names)]), get_classes(model)
        except Exception: pass
    if hasattr(model, 'predict_proba'):
        try: return model.predict_proba(val[feature_cols]), get_classes(model)
        except Exception: pass
    for front in (sel, pre):
        if front is None: continue
        try:
            fnames = getattr(front, 'feature_names_in_', None)
            Xf = val[list(fnames)] if fnames is not None and all(n in val.columns for n in fnames) else val[feature_cols]
            Z = as_frame(front.transform(Xf), front, val.index, feature_cols)
            mnames = getattr(model, 'feature_names_in_', None)
            if mnames is not None and all(n in Z.columns for n in mnames): Z = Z[list(mnames)]
            elif mnames is not None and len(mnames) == Z.shape[1]: Z = Z.copy(); Z.columns = list(mnames)
            return model.predict_proba(Z), get_classes(model)
        except Exception: pass
    try:
        pnames = getattr(pre, 'feature_names_in_', None)
        Xp = val[list(pnames)] if pnames is not None and all(n in val.columns for n in pnames) else val[feature_cols]
        Zp = as_frame(pre.transform(Xp), pre, val.index, feature_cols)
        snames = getattr(sel, 'feature_names_in_', None)
        if snames is not None and all(n in Zp.columns for n in snames): Zs = sel.transform(Zp[list(snames)])
        elif snames is not None and len(snames) == Zp.shape[1]: Zp2 = Zp.copy(); Zp2.columns = list(snames); Zs = sel.transform(Zp2)
        else: Zs = sel.transform(Zp)
        Zs = as_frame(Zs, sel, val.index)
        mnames = getattr(model, 'feature_names_in_', None)
        if mnames is not None and all(n in Zs.columns for n in mnames): Zs = Zs[list(mnames)]
        elif mnames is not None and len(mnames) == Zs.shape[1]: Zs = Zs.copy(); Zs.columns = list(mnames)
        return model.predict_proba(Zs), get_classes(model)
    except Exception as e:
        raise RuntimeError(f'could not score model: {e}')

rows = []; values = {}
for name, model in models.items():
    proba, classes = score_model(model)
    classes = classes or []
    if task_type == 'binary':
        pos = task.get('positive_label')
        if pos is None or pos not in classes: pos = classes[1]
        p = proba[:, classes.index(pos)]
        y = (val[target].to_numpy() == pos).astype(int)
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        fpr, tpr, thr = roc_curve(y, p); bi = int(np.argmax(tpr - fpr)); bt = float(thr[bi])
        ba = balanced_accuracy_score(y, (p >= bt).astype(int)); bs = brier_score_loss(y, p)
        rows += [(name, 'roc_auc', auc, np.nan), (name, 'pr_auc', ap, np.nan),
                 (name, 'balanced_accuracy', ba, bt), (name, 'brier_score', bs, np.nan)]
        values[name] = {'roc_auc': auc}
    else:
        y = val[target].to_numpy(); present = [c for c in classes if c in np.unique(y)]
        auc = float(np.mean([roc_auc_score((y == c).astype(int), proba[:, j]) for j, c in enumerate(classes) if c in present])) if present else np.nan
        pred = np.asarray(classes)[np.argmax(proba, axis=1)]
        f1 = f1_score(y, pred, average='macro')
        rows += [(name, 'roc_auc_ovr', auc, np.nan), (name, 'f1_macro', f1, np.nan)]
        values[name] = {'roc_auc_ovr': auc}

metric = primary if primary in next(iter(values.values())) else ('roc_auc' if task_type == 'binary' else 'roc_auc_ovr')
higher = metric not in ('brier_score',) and 'loss' not in metric
best_name = max(values, key=lambda n: values[n].get(metric, -np.inf)) if higher else min(values, key=lambda n: values[n].get(metric, np.inf))
df = pd.DataFrame(rows, columns=['model', 'metric', 'value', 'threshold'])
df['value'] = df['value'].round(4); df['threshold'] = df['threshold'].round(4)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics by candidate model')
for name, m in values.items():
    if primary in m: mlca_runtime.record_metric(primary, float(m[primary]), 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(values[best_name].get('roc_auc', np.nan)), source='validation_metrics.csv')
print(f"features={len(feature_cols)} models={len(models)} best={best_name}")
