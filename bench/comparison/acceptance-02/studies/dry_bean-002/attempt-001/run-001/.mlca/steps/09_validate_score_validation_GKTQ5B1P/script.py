import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, brier_score_loss, roc_curve, average_precision_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
if 'task' not in protocol:
    raise KeyError(f"Missing protocol key 'task'; available keys: {list(protocol)}")
task = protocol['task']
for k in ['targets', 'excluded_columns']:
    if k not in protocol:
        raise KeyError(f"Missing protocol key '{k}'; available keys: {list(protocol)}")
if 'primary_metric' not in task:
    raise KeyError(f"Missing task key 'primary_metric'; available task keys: {list(task)}")
primary = task['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
target = targets[0]
drop = set(targets) | set(excluded) | ({group} if group else set())

def keylike(s):
    if s.nunique(dropna=False) <= 1:
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return bool(s.is_unique)
    if pd.api.types.is_integer_dtype(s):
        return bool(s.is_unique)
    return False

feat_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]
raw_set = set(feat_cols)

def load(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

pre = load('05_preprocess_fit_preprocessing_MQJ4AQ8S/preprocessor.pkl')
sel = load('06_features_select_features_ZWFAMHT4/selector.pkl')

def in_names(o):
    if hasattr(o, 'feature_names_in_'):
        try:
            return list(o.feature_names_in_)
        except Exception:
            return None
    if hasattr(o, 'steps') and len(o.steps):
        return in_names(o.steps[0][1])
    return None

def get_classes(o):
    if hasattr(o, 'classes_'):
        return np.asarray(o.classes_)
    if hasattr(o, 'steps'):
        for _, s in reversed(o.steps):
            try:
                return get_classes(s)
            except Exception:
                pass
    if hasattr(o, 'named_steps'):
        for s in reversed(list(o.named_steps.values())):
            try:
                return get_classes(s)
            except Exception:
                pass
    raise AttributeError('classes_ not found')

def build(model):
    mn = in_names(model)
    if isinstance(model, Pipeline) and len(model.named_steps) > 1 and (mn is None or set(mn) <= raw_set):
        return model
    if mn is not None and set(mn) <= raw_set and hasattr(model, 'predict_proba'):
        return model
    sn, pn = in_names(sel), in_names(pre)
    steps = []
    if sn is not None and set(sn) <= raw_set:
        steps.append(('selector', sel))
    elif pn is not None and set(pn) <= raw_set:
        steps.append(('preprocessor', pre))
        if sel is not None:
            steps.append(('selector', sel))
    steps.append(('model', model))
    return steps[0][1] if len(steps) == 1 else Pipeline(steps)

def X_for(o):
    n = in_names(o)
    if n is not None:
        if set(n) & set(targets + excluded):
            raise ValueError('Saved model expects target or excluded column')
        return val[n]
    return val[feat_cols]

candidates = {
    'logistic': '07_model_train_logistic_G7CT94CC/model.pkl',
    'alternative': '08_model_train_alternative_PSET1F82/alternative_model.pkl',
}
rows, msgs = [], []
for name, key in candidates.items():
    model = build(load(key))
    Xv = X_for(model)
    proba = model.predict_proba(Xv)
    classes = get_classes(model)
    y = val[target].to_numpy()
    if len(classes) > 2:
        present = [c for c in classes if np.any(y == c)]
        left = [c for c in classes if c not in present]
        aucs = [roc_auc_score((y == c).astype(int), proba[:, j]) for j, c in enumerate(classes) if c in present]
        auc = float(np.mean(aucs)) if aucs else np.nan
        pred = model.predict(Xv)
        f1 = float(f1_score(y, pred, average='macro', labels=classes, zero_division=0))
        rows += [(name, primary, auc, np.nan), (name, 'f1_macro', f1, np.nan)]
        if left:
            msgs.append(f"{name} missing classes: {left}")
    else:
        pos = task.get('positive_label', classes[1])
        j = list(classes).index(pos)
        yb = (y == pos).astype(int)
        prob = proba[:, j]
        auc = roc_auc_score(yb, prob)
        pr = average_precision_score(yb, prob)
        fpr, tpr, thr = roc_curve(yb, prob)
        best_thr = float(thr[np.argmax(tpr - fpr)])
        bal = balanced_accuracy_score(yb, (prob >= best_thr).astype(int))
        brier = brier_score_loss(yb, prob)
        for m, v in [(primary, auc), ('pr_auc', pr), ('balanced_accuracy', bal), ('brier_score', brier)]:
            rows.append((name, m, float(v), best_thr))

df = pd.DataFrame(rows, columns=['model', 'metric', 'value', 'threshold'])
df['value'] = df['value'].map(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics for candidate models')
pri = df[df['metric'] == primary].copy()
lower_better = primary in {'brier_score', 'log_loss', 'rmse', 'mae'}
pri = pri.sort_values('value', ascending=lower_better)
for _, r in pri.iterrows():
    mlca_runtime.record_metric(primary, float(r['value']), split='val')
best = pri.iloc[-1] if lower_better else pri.iloc[-1]
mlca_runtime.record_fact_candidate('best_model', best['model'], source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best['value']), source='validation_metrics.csv')
print(f"train={train.shape} val={val.shape} features={len(feat_cols)}")
print(f"best_model={best['model']} {primary}={best['value']:.4g}")
if msgs:
    print('; '.join(msgs[:2]))
