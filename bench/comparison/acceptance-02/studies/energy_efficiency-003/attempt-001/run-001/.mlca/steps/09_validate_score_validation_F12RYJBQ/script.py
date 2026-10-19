import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
lower_is_better = primary_metric in ['rmse', 'mae', 'mse', 'log_loss']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feature_cols = [c for c in train.columns if c not in drop_cols]
for c in list(feature_cols):
    if (train[c].dtype == 'object' or pd.api.types.is_string_dtype(train[c])) and train[c].nunique() == len(train):
        feature_cols.remove(c)
    elif 'id' in c.lower() and train[c].nunique() == len(train):
        feature_cols.remove(c)

with open(inputs['05_preprocess_fit_preprocessing_H2H2H4ZR/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_3E3KFT0A/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_baseline_21R1ZBKQ/model.pkl'], 'rb') as f:
    model_baseline = pickle.load(f)
with open(inputs['08_model_train_alternative_QSEDYM24/alternative_model.pkl'], 'rb') as f:
    model_alt = pickle.load(f)

def is_complete_pipeline(m):
    if not isinstance(m, Pipeline):
        return False
    steps = [k.lower() for k in m.named_steps.keys()]
    kws = ['preprocess', 'select', 'scaler', 'encode', 'impute', 'transform', 'feature']
    return any(any(kw in s for kw in kws) for s in steps)

def build_pipeline(model, preprocessor, selector, raw_cols):
    if isinstance(model, Pipeline):
        if hasattr(model, 'feature_names_in_') and set(model.feature_names_in_) == set(raw_cols):
            return model
        if is_complete_pipeline(model):
            return model
    steps = []
    if isinstance(selector, Pipeline) and 'preprocessor' in selector.named_steps:
        steps.append(('frontend', selector))
    else:
        if preprocessor is not None:
            steps.append(('preprocessor', preprocessor))
        if selector is not None:
            steps.append(('selector', selector))
    steps.append(('model', model))
    return Pipeline(steps)

candidates = {
    'baseline': build_pipeline(model_baseline, preprocessor, selector, feature_cols),
    'alternative': build_pipeline(model_alt, preprocessor, selector, feature_cols)
}

def r4(x):
    return float(f"{x:.4g}")

rows = []
best_name, best_score = None, np.inf if lower_is_better else -np.inf
for name, pipe in candidates.items():
    X_val = val[feature_cols]
    y_pred = pipe.predict(X_val)
    rmses, maes, r2s = [], [], []
    for i, t in enumerate(targets):
        y_true = val[t].values
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        rmse = np.sqrt(mean_squared_error(y_true, yp))
        mae = mean_absolute_error(y_true, yp)
        r2 = r2_score(y_true, yp)
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
        rows += [{'model': name, 'metric': f'rmse_{t}', 'value': r4(rmse)},
                 {'model': name, 'metric': f'mae_{t}',  'value': r4(mae)},
                 {'model': name, 'metric': f'r2_{t}',   'value': r4(r2)}]
    macro_rmse, macro_mae, macro_r2 = np.mean(rmses), np.mean(maes), np.mean(r2s)
    rows += [{'model': name, 'metric': 'rmse_macro', 'value': r4(macro_rmse)},
             {'model': name, 'metric': 'mae_macro',  'value': r4(macro_mae)},
             {'model': name, 'metric': 'r2_macro',   'value': r4(macro_r2)}]
    if (lower_is_better and macro_rmse < best_score) or (not lower_is_better and macro_rmse > best_score):
        best_score, best_name = macro_rmse, name

mlca_runtime.save_table(pd.DataFrame(rows), 'validation_metrics.csv',
                        caption='Validation metrics per model and target')
mlca_runtime.record_metric(primary_metric, float(best_score), split='val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_validation_score', float(best_score), source='validation_metrics.csv')
print(f"Evaluated {len(candidates)} models on {len(val)} val rows")
print(f"Best model: {best_name}, macro {primary_metric}: {best_score:.4f}")
