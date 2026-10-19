import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
try:
    targets = list(proto['targets'])
    excluded = list(proto.get('excluded_columns', []))
    task = proto['task']
    primary = task['primary_metric']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto.keys())}")
group = proto.get('group_column')
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(train):
        continue
    feat_cols.append(c)
Xval = val[feat_cols].copy()
yval = val[targets[0]]

def load(key):
    with open(inp[key], 'rb') as f:
        return pickle.load(f)

pre = load('05_preprocess_fit_preprocessing_RMJEH56H/preprocessor.pkl')
sel = load('06_features_select_features_4X00PG4B/selector.pkl')
cands = {
    'logistic': load('07_model_train_logistic_7T5DR438/model.pkl'),
    'alternative': load('08_model_train_alternative_T24P90EK/alternative_model.pkl'),
}

def input_names(obj):
    try:
        if hasattr(obj, 'feature_names_in_'):
            return list(obj.feature_names_in_)
    except Exception:
        pass
    if isinstance(obj, Pipeline) and obj.steps:
        return input_names(obj.steps[0][1])
    return None

def accepts_raw(obj, raw):
    names = input_names(obj)
    return names is not None and set(names).issubset(set(raw))

def final_est(obj):
    return final_est(obj.steps[-1][1]) if isinstance(obj, Pipeline) else obj

def make_full(est):
    if isinstance(est, Pipeline):
        if 'preprocessor' in est.named_steps or 'selector' in est.named_steps:
            return est
        if accepts_raw(est, feat_cols):
            return est
    if accepts_raw(sel, feat_cols):
        return Pipeline([('selector', sel), ('model', est)])
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', est)])

def X_for(model, X):
    names = input_names(model)
    if names is None:
        return X
    missing = [c for c in names if c not in X.columns]
    if missing:
        raise KeyError(f"Missing input columns: {missing}")
    return X[names]

def macro_ovr(y, proba, classes):
    present = [c for c in classes if np.any(y == c)]
    scores = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        scores.append(roc_auc_score((y == c).astype(int), proba[:, j]))
    missing = [c for c in classes if c not in present]
    return float(np.mean(scores)) if scores else np.nan, missing

rows = []
for name, est in cands.items():
    model = make_full(est)
    X = X_for(model, Xval)
    proba = model.predict_proba(X)
    classes = np.asarray(final_est(model).classes_)
    pred = model.predict(X)
    auc, missing = macro_ovr(yval, proba, classes)
    f1 = f1_score(yval, pred, average='macro', labels=classes, zero_division=0)
    rows.append({'model': name, 'metric': primary, 'value': float(f"{auc:.4g}"), 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'f1_macro', 'value': float(f"{f1:.4g}"), 'threshold': np.nan})
    mlca_runtime.record_metric(primary, float(f"{auc:.4g}"), 'val')
    if missing:
        print(f"{name} missing classes: {missing}")

metrics = pd.DataFrame(rows)
mlca_runtime.save_table(metrics, 'validation_metrics.csv', caption='Validation metrics for candidate models')
lower_better = any(t in primary.lower() for t in ('brier', 'loss', 'error', 'mae', 'mse', 'rmse'))
prim = metrics[metrics['metric'] == primary]
best = prim.sort_values('value', ascending=lower_better).iloc[0]
mlca_runtime.record_fact_candidate('best_model', best['model'], source='validation_metrics.csv')
if 'roc_auc' in primary.lower():
    mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best['value']), source='validation_metrics.csv')
print(f"train={train.shape} val={val.shape} features={len(feat_cols)}")
print(f"best={best['model']} {primary}={best['value']}")
