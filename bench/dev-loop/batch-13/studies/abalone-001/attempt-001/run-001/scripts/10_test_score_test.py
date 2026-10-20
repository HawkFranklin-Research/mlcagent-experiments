import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
if 'group_column' not in protocol:
    raise KeyError(f"Missing 'group_column'. Available keys: {list(protocol.keys())}")
task = protocol['task']
primary_metric = task['primary_metric']
direction = {'rmse': 'min', 'mae': 'min', 'r2': 'max'}.get(primary_metric.lower(), 'min')

val_metrics = pd.read_csv(inputs['09_validate_score_validation_DX3NN439/validation_metrics.csv'])
def find_rows(df, metric):
    for col in (f'macro_{metric}', metric):
        sub = df[df['metric'] == col]
        if not sub.empty:
            return sub
    sub = df[df['metric'].astype(str).str.startswith(metric + '__')]
    if sub.empty:
        raise ValueError(f"No validation rows for '{metric}'. Available: {df['metric'].unique()[:20].tolist()}")
    return sub

cand = find_rows(val_metrics, primary_metric)
agg = cand.groupby('model')['value'].mean()
best_model_name = agg.idxmin() if direction == 'min' else agg.idxmax()
print(f"Candidate: {best_model_name} (val {primary_metric}={agg.loc[best_model_name]:.4f})")

if 'baseline' in str(best_model_name).lower():
    model_path = inputs['07_model_train_baseline_16VA3CWB/model.pkl']
elif 'alternative' in str(best_model_name).lower():
    model_path = inputs['08_model_train_alternative_YMQ80SAJ/alternative_model.pkl']
else:
    model_path = inputs['latest_model']
with open(model_path, 'rb') as f:
    model = pickle.load(f)

if not isinstance(model, Pipeline) and not hasattr(model, 'feature_names_in_'):
    with open(inputs['05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inputs['06_features_retain_features_W9C3S8F0/selector_all.pkl'], 'rb') as f:
        sel = pickle.load(f)
    model = Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)])

test = pd.read_parquet(inputs['test'])
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    train = pd.read_parquet(inputs['train'])
    def is_key_like(s):
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return s.nunique() == len(s)
        return pd.api.types.is_integer_dtype(s) and s.nunique() == len(s)
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded
                    and c != group_col and not is_key_like(train[c])]

X_test = test[feature_cols]
y_pred = np.asarray(model.predict(X_test))
y_true = test[targets].to_numpy()
if y_true.shape[1] == 1:
    y_true = y_true.ravel(); y_pred = y_pred.ravel()

metrics = []
for i, tgt in enumerate(targets):
    yt = y_true[:, i] if y_true.ndim > 1 else y_true
    yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred
    for m, v in (('rmse', np.sqrt(mean_squared_error(yt, yp))),
                 ('mae', mean_absolute_error(yt, yp)),
                 ('r2', r2_score(yt, yp))):
        metrics.append({'model': best_model_name, 'metric': f'{m}__{tgt}', 'value': round(float(v), 4)})

macro_vals = {}
for m in ('rmse', 'mae', 'r2'):
    vals = [r['value'] for r in metrics if r['metric'].startswith(m + '__')]
    macro_vals[m] = round(float(np.mean(vals)), 4)
    metrics.append({'model': best_model_name, 'metric': f'macro_{m}', 'value': macro_vals[m]})

mlca_runtime.save_table(pd.DataFrame(metrics), "test_metrics.csv", caption="Test metrics per target and macro averages")

primary_value = macro_vals[primary_metric]
mlca_runtime.record_metric(primary_metric, primary_value, split='test')
mlca_runtime.record_fact_candidate('test_score', primary_value, source='test_metrics.csv')

pred_df = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
if y_true.ndim == 1:
    pred_df[f'y_true__{targets[0]}'] = y_true
    pred_df[f'y_pred__{targets[0]}'] = y_pred
else:
    for i, tgt in enumerate(targets):
        pred_df[f'y_true__{tgt}'] = y_true[:, i]
        pred_df[f'y_pred__{tgt}'] = y_pred[:, i]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")
print(f"Test {primary_metric}: {primary_value:.4f}")
