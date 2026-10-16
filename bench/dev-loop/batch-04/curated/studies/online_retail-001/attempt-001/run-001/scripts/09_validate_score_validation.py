import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary = task['primary_metric']
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)

val = pd.read_parquet(inputs['val'])
feature_cols = [c for c in val.columns if c not in targets and c not in excluded and c != group]

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_input_columns(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

def apply_frontend(X, pre, sel):
    if isinstance(sel, Pipeline):
        return sel.transform(X)
    Xt = X
    if pre is not None:
        Xt = pre.transform(Xt)
    if sel is not None:
        Xt = sel.transform(Xt)
    return Xt

candidates = {
    'baseline': inputs['07_model_train_baseline_DWY924K5/model.pkl'],
    'alternative': inputs['08_model_train_alternative_3C755C37/alternative_model.pkl'],
}

pre = load_pkl(inputs['05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl'])
sel = load_pkl(inputs['06_features_retain_features_RHW0T9SM/selector_all.pkl'])

rows = []
for name, path in candidates.items():
    obj = load_pkl(path)
    model = obj
    if isinstance(obj, dict):
        model = obj.get('model', obj.get('estimator', obj.get('pipeline', obj)))
    if isinstance(model, Pipeline):
        expected = get_input_columns(model)
        X_val = val[expected].copy() if expected is not None else val[feature_cols].copy()
        labels = model.predict(X_val)
        X_trans = model[:-1].transform(X_val) if len(model.steps) > 1 else X_val
    else:
        expected = get_input_columns(pre) or get_input_columns(sel) or get_input_columns(model)
        X_val = val[expected].copy() if expected is not None else val[feature_cols].copy()
        X_trans = apply_frontend(X_val, pre, sel)
        labels = model.predict(X_trans)
    labels = np.asarray(labels).ravel()
    X_trans = X_trans.toarray() if hasattr(X_trans, 'toarray') else np.asarray(X_trans)
    n = len(X_trans)
    if n > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=10000, replace=False)
        X_metric = X_trans[idx]
        labels_metric = labels[idx]
        print(f'{name}: subsampled {len(idx)} of {n} for metrics')
    else:
        X_metric = X_trans
        labels_metric = labels
    if len(np.unique(labels_metric)) < 2:
        sil = ch = db = np.nan
    else:
        sil = silhouette_score(X_metric, labels_metric, sample_size=min(10000, len(labels_metric)), random_state=42)
        ch = calinski_harabasz_score(X_metric, labels_metric)
        db = davies_bouldin_score(X_metric, labels_metric)
    rows.append({'model': name, 'metric': 'silhouette', 'value': sil})
    rows.append({'model': name, 'metric': 'calinski_harabasz', 'value': ch})
    rows.append({'model': name, 'metric': 'davies_bouldin', 'value': db})

df = pd.DataFrame(rows)
df['value'] = df['value'].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation clustering metrics')

metric_map = {
    'silhouette': 'silhouette',
    'calinski_harabasz': 'calinski_harabasz',
    'calinski-harabasz': 'calinski_harabasz',
    'davies_bouldin': 'davies_bouldin',
    'davies-bouldin': 'davies_bouldin',
}
primary_key = metric_map.get(primary, primary)
primary_rows = df[df['metric'] == primary_key]
if primary_rows.empty:
    raise ValueError(f'Primary metric {primary} not computed')
if primary_key in ['silhouette', 'calinski_harabasz']:
    best_row = primary_rows.loc[primary_rows['value'].idxmax()]
else:
    best_row = primary_rows.loc[primary_rows['value'].idxmin()]
best_model = best_row['model']
best_score = float(best_row['value'])
mlca_runtime.record_metric(primary, best_score, 'val')
mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')

print(f'val rows: {len(val)}')
print(f'candidates: {len(candidates)}')
print(f'best {primary}: {best_model} = {best_score:.4g}')
