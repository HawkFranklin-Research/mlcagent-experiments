import mlca_runtime
import json, pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
primary_metric = task['primary_metric']

minimize = primary_metric.lower() in {'rmse','mae','mse','mape','logloss','mean_squared_error','mean_absolute_error'}

val_metrics = pd.read_csv(inputs['09_validate_score_validation_1WV4GQNX/validation_metrics.csv'])
if 'split' in val_metrics.columns:
    val_metrics = val_metrics[val_metrics['split'] == 'val']
if 'metric' in val_metrics.columns:
    val_metrics = val_metrics[val_metrics['metric'] == primary_metric]
model_scores = val_metrics.groupby('model')['value'].mean().reset_index()
if minimize:
    best_model = model_scores.loc[model_scores['value'].idxmin(), 'model']
else:
    best_model = model_scores.loc[model_scores['value'].idxmax(), 'model']

if 'baseline' in str(best_model).lower():
    model_path = inputs['07_model_train_baseline_3H582954/model.pkl']
elif 'alternative' in str(best_model).lower() or 'latest' in str(best_model).lower():
    model_path = inputs['08_model_train_alternative_J64AD4FF/alternative_model.pkl']
else:
    model_path = inputs['latest_model']
with open(model_path, 'rb') as f:
    model = pickle.load(f)

test_df = pd.read_parquet(inputs['test'])
feature_cols = None
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
elif hasattr(model, 'steps'):
    for _, step in model.steps:
        if hasattr(step, 'feature_names_in_'):
            feature_cols = list(step.feature_names_in_)
            break
if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = []
    for c in test_df.columns:
        if c in drop:
            continue
        s = test_df[c]
        if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == len(s):
            continue
        feature_cols.append(c)

X_test = test_df[feature_cols]
y_test = test_df[targets].values
y_pred = model.predict(X_test)
if y_pred.ndim == 1 and len(targets) == 1:
    y_pred = y_pred.reshape(-1, 1)
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)

rows = []
for i, tgt in enumerate(targets):
    yt, yp = y_test[:, i], y_pred[:, i]
    metrics = {
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'mae': mean_absolute_error(yt, yp),
        'r2': r2_score(yt, yp),
    }
    for name, val in metrics.items():
        rows.append({'model': best_model, 'metric': f'{name}_{tgt}', 'value': val})

for name in ['rmse', 'mae', 'r2']:
    vals = [r['value'] for r in rows if r['metric'].startswith(name + '_')]
    rows.append({'model': best_model, 'metric': f'{name}_macro', 'value': float(np.mean(vals))})

for r in rows:
    r['value'] = float(f"{r['value']:.4g}") if not pd.isna(r['value']) else r['value']

test_metrics = pd.DataFrame(rows)
mlca_runtime.save_table(test_metrics, "test_metrics.csv", caption="Test metrics for validation-selected model")

primary_row = test_metrics[test_metrics['metric'] == f'{primary_metric}_macro']
primary_value = float(primary_row.iloc[0]['value']) if len(primary_row) else float(test_metrics[test_metrics['metric'].str.startswith(primary_metric)]['value'].mean())
mlca_runtime.record_metric(primary_metric, primary_value, split='test')
mlca_runtime.record_fact_candidate('test_score', primary_value, source='test_metrics.csv')

print(f"Selected model: {best_model}")
print(f"Test rows: {len(test_df)}, features: {len(feature_cols)}")
print(f"{primary_metric} macro test: {primary_value:.4g}")
print("Saved test_metrics.csv")
