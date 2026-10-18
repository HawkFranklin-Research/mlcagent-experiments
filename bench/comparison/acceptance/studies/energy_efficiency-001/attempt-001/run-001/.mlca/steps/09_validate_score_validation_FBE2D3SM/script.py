import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
outdir = mlca_runtime.output_dir()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']
direction = 'min' if primary_metric in ('rmse', 'mae') else 'max'

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and (group is None or c != group)]
key_like = []
for c in feat_cols:
    s = train[c]
    if s.dtype.kind in 'iO' and s.nunique() == len(s) and len(s) > 1:
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]
X_val = val[feat_cols].copy()

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

preproc = load_pkl(inputs['05_preprocess_fit_preprocessing_Y0BA94GV/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_retain_features_J9XJJ7CH/selector_all.pkl'])
baseline_model = load_pkl(inputs['07_model_train_baseline_RHMAVZ3Z/model.pkl'])
alt_model = load_pkl(inputs['08_model_train_alternative_PZKEGXY4/alternative_model.pkl'])

def make_pipeline(model, preproc, selector):
    if isinstance(model, Pipeline):
        if len(model.named_steps) > 1:
            return model
        model = list(model.named_steps.values())[0]
    if isinstance(selector, Pipeline):
        if preproc is not None and any(step is preproc for step in selector.named_steps.values()):
            return Pipeline(list(selector.steps) + [('model', model)])
        return Pipeline([('preprocessor', preproc), ('selector', selector), ('model', model)])
    return Pipeline([('preprocessor', preproc), ('selector', selector), ('model', model)])

candidates = {'baseline': baseline_model, 'alternative': alt_model}
results = []

for name, model in candidates.items():
    pipe = make_pipeline(model, preproc, selector)
    y_pred = pipe.predict(X_val)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    y_val = val[targets].values
    for i, t in enumerate(targets):
        yt, yp = y_val[:, i], y_pred[:, i]
        results.append({'model': name, 'metric': f'{t}_rmse', 'value': np.sqrt(mean_squared_error(yt, yp))})
        results.append({'model': name, 'metric': f'{t}_mae', 'value': mean_absolute_error(yt, yp)})
        results.append({'model': name, 'metric': f'{t}_r2', 'value': r2_score(yt, yp)})
    rmse_macro = np.mean([np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])) for i in range(len(targets))])
    mae_macro = np.mean([mean_absolute_error(y_val[:, i], y_pred[:, i]) for i in range(len(targets))])
    r2_macro = np.mean([r2_score(y_val[:, i], y_pred[:, i]) for i in range(len(targets))])
    results.append({'model': name, 'metric': 'macro_rmse', 'value': rmse_macro})
    results.append({'model': name, 'metric': 'macro_mae', 'value': mae_macro})
    results.append({'model': name, 'metric': 'macro_r2', 'value': r2_macro})

metrics_df = pd.DataFrame(results)
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv', caption='Validation metrics per model and target')

prim_rows = metrics_df[metrics_df['metric'] == f'macro_{primary_metric}']
if direction == 'min':
    best_row = prim_rows.loc[prim_rows['value'].idxmin()]
else:
    best_row = prim_rows.loc[prim_rows['value'].idxmax()]
best_model_name = best_row['model']
best_value = float(best_row['value'])

mlca_runtime.record_metric(primary_metric, best_value, split='val')
mlca_runtime.record_fact_candidate('best_model', best_model_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_validation_score', best_value, source='validation_metrics.csv')

print(f"Validation metrics saved. Best model: {best_model_name} with {primary_metric}={best_value:.4f}")
