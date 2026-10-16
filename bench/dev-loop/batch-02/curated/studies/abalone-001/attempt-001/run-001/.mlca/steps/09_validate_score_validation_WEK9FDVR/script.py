import mlca_runtime
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group = protocol['group_column']
    task = protocol['task']
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available top-level keys: {list(protocol.keys())}")

primary_metric = task.get('primary_metric') or protocol.get('primary_metric') or 'rmse'

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]
for c in list(feature_cols):
    if train[c].dtype == 'object' or train[c].dtype == 'string':
        if train[c].nunique() == len(train):
            feature_cols.remove(c)
    elif train[c].dtype.kind in 'iu':
        if train[c].nunique() == len(train):
            feature_cols.remove(c)

def load_pkl(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl('05_preprocess_fit_preprocessing_JT5JG4F1/preprocessor.pkl')
selector = load_pkl('06_features_retain_features_6CDSH3YD/selector_all.pkl')
model_baseline = load_pkl('07_model_train_baseline_F3SAKS84/model.pkl')
model_alt = load_pkl('08_model_train_alternative_G7VTXE7A/alternative_model.pkl')
model_latest = load_pkl('latest_model')

def build_pipeline(pre, sel, mod, feature_cols):
    if isinstance(mod, Pipeline):
        first_step = list(mod.named_steps.values())[0]
        if hasattr(first_step, 'feature_names_in_') and list(first_step.feature_names_in_) == feature_cols:
            return mod
        step_names = list(mod.named_steps.keys())
        has_pre = any('preprocess' in name.lower() for name in step_names)
        has_sel = any('select' in name.lower() for name in step_names)
        if has_pre and has_sel:
            return mod
        elif has_sel:
            return Pipeline([('preprocessor', pre), ('model', mod)])
        else:
            return Pipeline([('preprocessor', pre), ('selector', sel), ('model', mod)])
    else:
        if isinstance(sel, Pipeline):
            first_step = list(sel.named_steps.values())[0]
            if hasattr(first_step, 'feature_names_in_') and list(first_step.feature_names_in_) == feature_cols:
                return Pipeline([('selector', sel), ('model', mod)])
            step_names = list(sel.named_steps.keys())
            has_pre = any('preprocess' in name.lower() for name in step_names)
            if has_pre:
                return Pipeline([('selector', sel), ('model', mod)])
            else:
                return Pipeline([('preprocessor', pre), ('selector', sel), ('model', mod)])
        else:
            return Pipeline([('preprocessor', pre), ('selector', sel), ('model', mod)])

candidates = {
    'baseline': build_pipeline(preprocessor, selector, model_baseline, feature_cols),
    'alternative': build_pipeline(preprocessor, selector, model_alt, feature_cols),
    'latest': build_pipeline(preprocessor, selector, model_latest, feature_cols)
}

X_val = val[feature_cols]
results = []
for name, pipe in candidates.items():
    y_pred = pipe.predict(X_val)
    if len(targets) == 1:
        y_pred = y_pred.reshape(-1, 1)
    y_true = val[targets].values
    for i, target in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        results.append({'model': name, 'metric': f'rmse_{target}', 'value': rmse})
        results.append({'model': name, 'metric': f'mae_{target}', 'value': mae})
        results.append({'model': name, 'metric': f'r2_{target}', 'value': r2})
    rmse_macro = np.mean([r['value'] for r in results if r['model'] == name and r['metric'].startswith('rmse_')])
    mae_macro = np.mean([r['value'] for r in results if r['model'] == name and r['metric'].startswith('mae_')])
    r2_macro = np.mean([r['value'] for r in results if r['model'] == name and r['metric'].startswith('r2_')])
    results.append({'model': name, 'metric': 'rmse_macro', 'value': rmse_macro})
    results.append({'model': name, 'metric': 'mae_macro', 'value': mae_macro})
    results.append({'model': name, 'metric': 'r2_macro', 'value': r2_macro})

df_results = pd.DataFrame(results)
df_results['value'] = df_results['value'].apply(lambda x: float(f'{x:.4g}'))
mlca_runtime.save_table(df_results, 'validation_metrics.csv', caption='Validation metrics per model and target')

best_row = df_results[df_results['metric'] == 'rmse_macro'].sort_values('value').iloc[0]
best_model = best_row['model']
best_score = best_row['value']

mlca_runtime.record_metric('rmse', best_score, 'val')
mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')

print(f"Evaluated {len(candidates)} models on val. Best: {best_model} with rmse_macro={best_score:.4g}")
