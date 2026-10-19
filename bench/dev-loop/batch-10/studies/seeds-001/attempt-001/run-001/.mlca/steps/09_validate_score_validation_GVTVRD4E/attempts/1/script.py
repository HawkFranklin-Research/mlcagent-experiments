import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)

train_df = pd.read_parquet(inputs['train'])
val_df = pd.read_parquet(inputs['val'])

def is_identifier(series):
    if series.dtype.kind in 'OSU':
        return series.nunique() == len(series)
    if series.dtype.kind == 'i':
        return series.nunique() == len(series)
    return False

feature_cols = [c for c in train_df.columns
                if c not in targets and c not in excluded and c != group
                and not is_identifier(train_df[c])]
X_val = val_df[feature_cols].copy()

with open(inputs['05_preprocess_fit_preprocessing_6DN7THMT/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)
with open(inputs['06_features_retain_features_AKTAEF6F/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_baseline_V7ETXH3B/model.pkl'], 'rb') as f:
    baseline_obj = pickle.load(f)
with open(inputs['08_model_train_alternative_G1KJG23J/alternative_model.pkl'], 'rb') as f:
    alt_obj = pickle.load(f)

def prepend_step(steps, name, transformer):
    existing = [n for n, _ in steps]
    new_name = name
    i = 1
    while new_name in existing:
        new_name = f"{name}_{i}"
        i += 1
    return [(new_name, transformer)] + list(steps)

def get_frontend_and_final(model_obj, preproc, selector, feature_cols):
    if isinstance(model_obj, Pipeline):
        first = model_obj.steps[0][1]
        if hasattr(first, 'feature_names_in_') and set(first.feature_names_in_) == set(feature_cols):
            frontend = Pipeline(model_obj.steps[:-1]) if len(model_obj.steps) > 1 else None
            final = model_obj.steps[-1][1]
            return frontend, final
        else:
            steps = prepend_step(model_obj.steps[:-1], 'preproc', preproc)
            frontend = Pipeline(steps) if steps else None
            final = model_obj.steps[-1][1]
            return frontend, final
    else:
        if selector is not None and isinstance(selector, Pipeline) and hasattr(selector, 'feature_names_in_'):
            if set(selector.feature_names_in_) == set(feature_cols):
                frontend = selector
            else:
                steps = prepend_step(selector.steps, 'preproc', preproc)
                frontend = Pipeline(steps) if steps else None
        elif selector is not None:
            frontend = Pipeline([('preproc', preproc), ('selector', selector)])
        else:
            frontend = preproc
        return frontend, model_obj

def reconstruction_error(model_obj, preproc, selector, X_raw, feature_cols):
    frontend, final = get_frontend_and_final(model_obj, preproc, selector, feature_cols)
    if frontend is None:
        Xf = X_raw
    else:
        if hasattr(frontend, 'feature_names_in_'):
            cols = list(frontend.feature_names_in_)
            X_in = X_raw[cols]
        else:
            X_in = X_raw
        Xf = frontend.transform(X_in)
    if hasattr(Xf, 'values'):
        Xf = Xf.values
    if hasattr(final, 'transform') and hasattr(final, 'inverse_transform'):
        recon = final.inverse_transform(final.transform(Xf))
    elif hasattr(final, 'predict'):
        recon = final.predict(Xf)
    else:
        raise ValueError(f"Unknown estimator {type(final)}")
    return float(np.mean((Xf - recon) ** 2))

mse_baseline = reconstruction_error(baseline_obj, preproc, selector, X_val, feature_cols)
mse_alt = reconstruction_error(alt_obj, preproc, selector, X_val, feature_cols)

cands = {'baseline': mse_baseline, 'alternative': mse_alt}
best_model = min(cands, key=cands.get)
best_score = cands[best_model]

metrics_df = pd.DataFrame([
    {'model': 'baseline', 'metric': primary_metric, 'value': mse_baseline},
    {'model': 'alternative', 'metric': primary_metric, 'value': mse_alt},
])
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv', caption='Validation reconstruction errors')

mlca_runtime.record_metric(primary_metric, best_score, 'val')
mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')

print(f"baseline: {mse_baseline:.4f}")
print(f"alternative: {mse_alt:.4f}")
print(f"best: {best_model} {best_score:.4f}")
