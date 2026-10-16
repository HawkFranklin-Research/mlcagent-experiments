import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
primary_metric = protocol['task']['primary_metric']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

val = pd.read_parquet(inputs['val'])

preprocessor = pickle.load(open(inputs['05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl'], 'rb'))
selector = pickle.load(open(inputs['06_features_retain_features_P9FZ0GCB/selector_all.pkl'], 'rb'))

def get_input_columns(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if hasattr(obj, 'named_steps'):
        first_step = list(obj.named_steps.values())[0]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    if hasattr(obj, 'steps'):
        first_step = obj.steps[0][1]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    return None

def get_encoders(obj):
    for attr in ['encoders', 'label_encoders', 'target_encoders']:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    if hasattr(obj, 'named_steps'):
        for step in obj.named_steps.values():
            enc = get_encoders(step)
            if enc is not None:
                return enc
    if hasattr(obj, 'steps'):
        for _, step in obj.steps:
            enc = get_encoders(step)
            if enc is not None:
                return enc
    return None

def get_pipeline(model):
    if isinstance(model, Pipeline):
        return model
    return Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])

candidates = {
    'baseline': pickle.load(open(inputs['07_model_train_baseline_SGXK026F/model.pkl'], 'rb')),
    'alternative': pickle.load(open(inputs['08_model_train_alternative_MK5K74KS/alternative_model.pkl'], 'rb'))
}

results = []
for name, model in candidates.items():
    pipe = get_pipeline(model)
    input_cols = get_input_columns(pipe)
    if input_cols is None:
        all_cols = val.columns.tolist()
        input_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
        input_cols = [c for c in input_cols if not (val[c].dtype == 'object' and val[c].nunique() == len(val))]
    missing = [c for c in input_cols if c not in val.columns]
    if missing:
        raise ValueError(f"Missing columns for {name}: {missing}")
    X_val = val[input_cols]
    y_pred = pipe.predict(X_val)
    if isinstance(y_pred, list):
        y_pred = np.column_stack(y_pred)
    elif hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    y_true = val[targets].values
    encoders = get_encoders(pipe)
    per_f1, per_hl = [], []
    y_true_enc, y_pred_enc = [], []
    for i, tname in enumerate(targets):
        yt = y_true[:, i]
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        if yp.dtype.kind in 'iu' and yt.dtype.kind == 'O':
            if encoders is not None:
                enc = encoders[i] if isinstance(encoders, list) else encoders.get(tname)
                if enc is not None:
                    yt = enc.transform(yt)
                else:
                    raise ValueError(f"No encoder for {tname}")
            else:
                raise ValueError(f"No encoders for {tname}")
        per_f1.append(f1_score(yt, yp, average='macro'))
        per_hl.append(hamming_loss(yt, yp))
        y_true_enc.append(yt)
        y_pred_enc.append(yp)
    mean_f1 = float(np.mean(per_f1))
    mean_hl = float(np.mean(per_hl))
    subset_acc = float(np.mean(np.all(np.column_stack(y_true_enc) == np.column_stack(y_pred_enc), axis=1)))
    results.append({
        'model': name,
        'macro_f1': mean_f1,
        'hamming_loss': mean_hl,
        'subset_accuracy': subset_acc,
        'per_target_f1': per_f1,
        'per_target_hamming': per_hl
    })

rows = []
for res in results:
    m = res['model']
    rows.append({'model': m, 'metric': 'macro_f1', 'value': res['macro_f1']})
    rows.append({'model': m, 'metric': 'hamming_loss', 'value': res['hamming_loss']})
    rows.append({'model': m, 'metric': 'subset_accuracy', 'value': res['subset_accuracy']})
    for i, tname in enumerate(targets):
        rows.append({'model': m, 'metric': f'macro_f1_{tname}', 'value': res['per_target_f1'][i]})
        rows.append({'model': m, 'metric': f'hamming_loss_{tname}', 'value': res['per_target_hamming'][i]})

df_metrics = pd.DataFrame(rows)
df_metrics['value'] = df_metrics['value'].round(4)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics for candidate models')

best = max(results, key=lambda x: x['macro_f1'])
mlca_runtime.record_metric(primary_metric, best['macro_f1'], split='val')
mlca_runtime.record_fact_candidate('best_model', best['model'], source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_validation_score', best['macro_f1'], source='validation_metrics.csv')

print(f"Val shape: {val.shape}")
print(f"Best model: {best['model']}, {primary_metric}: {best['macro_f1']:.4f}")
