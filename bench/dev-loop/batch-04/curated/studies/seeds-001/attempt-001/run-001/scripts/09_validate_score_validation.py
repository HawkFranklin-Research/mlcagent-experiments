import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")
task = protocol['task']
if 'primary_metric' not in task:
    raise KeyError(f"Missing primary_metric in task. Available: {list(task.keys())}")
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

val = pd.read_parquet(inputs['val'])
n_val = len(val)
sample_size = min(10000, n_val)

model_paths = {
    'baseline': inputs['07_model_train_baseline_BPBM7F9H/model.pkl'],
    'alternative': inputs['08_model_train_alternative_09MZ25S1/alternative_model.pkl'],
}

def get_feature_cols(model, df):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
    keep = []
    for c in cols:
        s = df[c]
        if s.dtype == object or str(s.dtype) == 'string':
            if s.nunique() == len(s):
                continue
        if pd.api.types.is_integer_dtype(s):
            if s.nunique() == len(s):
                continue
        keep.append(c)
    return keep

rows = []
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        model = pickle.load(f)
    feat_cols = get_feature_cols(model, val)
    missing = [c for c in feat_cols if c not in val.columns]
    if missing:
        raise ValueError(f"Missing features for {name}: {missing}")
    X_val = val[feat_cols]
    labels = model.predict(X_val)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= n_val:
        sil = np.nan
    else:
        sil = silhouette_score(X_val, labels, sample_size=sample_size, random_state=42)
    ch = calinski_harabasz_score(X_val, labels) if len(unique) >= 2 else np.nan
    db = davies_bouldin_score(X_val, labels) if len(unique) >= 2 else np.nan
    rows.append({'model': name, 'metric': 'silhouette', 'value': sil})
    rows.append({'model': name, 'metric': 'calinski_harabasz', 'value': ch})
    rows.append({'model': name, 'metric': 'davies_bouldin', 'value': db})

df_metrics = pd.DataFrame(rows)
df_metrics['value'] = df_metrics['value'].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df_metrics, "validation_metrics.csv", caption="Validation metrics for clustering candidates")

direction = {'silhouette': 'max', 'calinski_harabasz': 'max', 'davies_bouldin': 'min'}
if primary_metric not in direction:
    raise ValueError(f"Unknown primary_metric: {primary_metric}")
prim = df_metrics[df_metrics['metric'] == primary_metric].dropna(subset=['value'])
if prim.empty:
    raise ValueError(f"No valid {primary_metric} scores.")
if direction[primary_metric] == 'max':
    best_row = prim.loc[prim['value'].idxmax()]
else:
    best_row = prim.loc[prim['value'].idxmin()]
best_model = best_row['model']
best_score = float(best_row['value'])

mlca_runtime.record_metric(primary_metric, best_score, split='val')
mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')

print(f"Val rows: {n_val}, silhouette sample_size: {sample_size}")
print(f"Primary metric: {primary_metric}")
print(f"Best model: {best_model} with {primary_metric}={best_score:.4g}")
