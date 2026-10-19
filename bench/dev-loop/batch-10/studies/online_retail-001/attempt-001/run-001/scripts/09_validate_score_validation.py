import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
primary_metric = protocol['task']['primary_metric']
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

val = pd.read_parquet(inputs['val'])

model_paths = {
    'baseline': inputs['07_model_train_baseline_726ZEXFH/model.pkl'],
    'alternative': inputs['08_model_train_alternative_R61594N7/alternative_model.pkl']
}

records = []
best_score = None
best_model_name = None
higher_better = primary_metric in ('silhouette', 'calinski_harabasz')

for name, path in model_paths.items():
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
    else:
        raise ValueError(f"Model {name} lacks feature_names_in_")
    X_val = val[cols]
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        Xt = model[:-1].transform(X_val)
    else:
        Xt = X_val.values if hasattr(X_val, 'values') else X_val
    labels = model.predict(X_val)
    n_samples = Xt.shape[0]
    sample_size = min(10000, n_samples) if n_samples > 10000 else None
    for metric in ['silhouette', 'calinski_harabasz', 'davies_bouldin']:
        try:
            if metric == 'silhouette':
                val_metric = silhouette_score(Xt, labels, sample_size=sample_size, random_state=42)
            elif metric == 'calinski_harabasz':
                val_metric = calinski_harabasz_score(Xt, labels)
            else:
                val_metric = davies_bouldin_score(Xt, labels)
        except Exception:
            val_metric = np.nan
        records.append({'model': name, 'metric': metric, 'value': val_metric})

df_metrics = pd.DataFrame(records)
df_metrics['value'] = df_metrics['value'].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation clustering metrics')

primary_df = df_metrics[df_metrics['metric'] == primary_metric].dropna(subset=['value'])
if not primary_df.empty:
    if higher_better:
        best_row = primary_df.loc[primary_df['value'].idxmax()]
    else:
        best_row = primary_df.loc[primary_df['value'].idxmin()]
    best_model_name = best_row['model']
    best_score = best_row['value']
    mlca_runtime.record_metric(primary_metric, best_score, split='val')
    mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')
    mlca_runtime.record_fact_candidate('best_model', best_model_name, source='validation_metrics.csv')

print(f"Validation rows: {len(val)}")
print(f"Best {primary_metric}: {best_model_name} = {best_score}")
