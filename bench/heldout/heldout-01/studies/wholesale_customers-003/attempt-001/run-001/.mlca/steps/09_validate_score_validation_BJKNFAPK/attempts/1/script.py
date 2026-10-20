import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp['_protocol']).read_text())
primary = protocol['task']['primary_metric']
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inp['05_preprocess_fit_preprocessing_TEP20BJ6/preprocessor.pkl'])
sel = load_pkl(inp['06_features_retain_features_HRWJ3W8S/selector_all.pkl'])
models = {
    'baseline_model': load_pkl(inp['07_model_train_baseline_4QYJGXS4/model.pkl']),
    'alternative_model': load_pkl(inp['latest_model']),
}

def input_cols(m):
    for o in (m, sel, pre):
        fn = getattr(o, 'feature_names_in_', None) if o is not None else None
        if fn is not None:
            return list(fn)
    return list(val.columns)

def cluster_space(m, X):
    if isinstance(m, Pipeline) and len(m.steps) > 1:
        return np.asarray(Pipeline(m.steps[:-1]).transform(X))
    Z = X
    if pre is not None:
        pin = getattr(pre, 'feature_names_in_', None)
        Z = pre.transform(Z[list(pin)]) if pin is not None else pre.transform(Z)
    if sel is not None and isinstance(Z, pd.DataFrame):
        sin = getattr(sel, 'feature_names_in_', None)
        Z = sel.transform(Z[list(sin)]) if sin is not None else sel.transform(Z)
    return np.asarray(Z)

rng = np.random.RandomState(42)
rows, best_name, best_score = [], None, -np.inf
for name, m in models.items():
    cols = input_cols(m)
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"val missing columns: {missing}")
    X = val[cols].copy()
    labels = np.asarray(m.predict(X))
    Z = cluster_space(m, X)
    n = min(10000, Z.shape[0])
    if n < Z.shape[0]:
        idx = rng.choice(Z.shape[0], n, replace=False)
        Zs, ls = Z[idx], labels[idx]
    else:
        Zs, ls = Z, labels
    k = len(np.unique(ls))
    if k < 2 or k >= Zs.shape[0]:
        sil = ch = db = float('nan')
    else:
        sil = float(silhouette_score(Zs, ls, sample_size=n, random_state=42))
        ch = float(calinski_harabasz_score(Zs, ls))
        db = float(davies_bouldin_score(Zs, ls))
    rows += [
        {'model': name, 'metric': 'silhouette', 'value': sil},
        {'model': name, 'metric': 'calinski_harabasz', 'value': ch},
        {'model': name, 'metric': 'davies_bouldin', 'value': db},
    ]
    score = sil if not np.isnan(sil) else -np.inf
    if score > best_score:
        best_score, best_name = score, name

metrics_df = pd.DataFrame(rows)
metrics_df['value'] = pd.to_numeric(metrics_df['value'], errors='coerce').round(4)
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv',
                        caption='Validation clustering metrics per saved candidate.')
best_val = float(metrics_df[(metrics_df['model'] == best_name) &
                            (metrics_df['metric'] == primary)]['value'].iloc[0])
mlca_runtime.record_metric(primary, best_val, 'val')
mlca_runtime.record_fact_candidate('best_validation_score', best_val,
                                   source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_name,
                                   source='validation_metrics.csv')
mlca_runtime.save_model(models[best_name], 'frozen_candidate')
print(f"candidates={list(models)} best={best_name} {primary}={best_val:.4f}")
