import mlca_runtime, json, pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def load_json(key):
    return json.loads(inputs[key].read_text())

def load_pkl(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

protocol = load_json('_protocol')
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k} in _protocol. Available: {list(protocol.keys())}")
targets = protocol['targets']
task = protocol['task']
metric = task['primary_metric']

baseline = load_pkl('07_model_train_baseline_3H582954/model.pkl')
alternative = load_pkl('08_model_train_alternative_J64AD4FF/alternative_model.pkl')
val_metrics = pd.read_csv(inputs['09_validate_score_validation_1WV4GQNX/validation_metrics.csv'])
pre_manifest = load_json('05_preprocess_fit_preprocessing_QJRR3HF7/manifest.json')
sel_manifest = load_json('06_features_retain_features_8C5RPJ11/manifest.json')

def key_params(est):
    keys = ['n_estimators','max_depth','learning_rate','alpha','C','penalty','solver','loss','min_samples_leaf','max_iter']
    out = [f"{k}={getattr(est,k)}" for k in keys if hasattr(est,k)]
    return ', '.join(out) if out else 'default'

def model_info(model):
    if hasattr(model, 'named_steps'):
        steps = list(model.named_steps.keys())
        est = model.named_steps[steps[-1]]
        feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
    else:
        est = model
        feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
    if not feats and hasattr(est, 'feature_names_in_'):
        feats = list(est.feature_names_in_)
    top = 'unknown'
    if feats:
        if hasattr(est, 'coef_'):
            c = np.ravel(est.coef_)
            if len(c) == len(feats): top = feats[int(np.argmax(np.abs(c)))]
        elif hasattr(est, 'feature_importances_'):
            imp = est.feature_importances_
            if len(imp) == len(feats): top = feats[int(np.argmax(imp))]
    return {'class': type(model).__name__, 'est': type(est).__name__,
            'params': key_params(est), 'features': feats, 'n': len(feats), 'top': top}

bi = model_info(baseline)
ai = model_info(alternative)
top = bi['top'] if bi['top'] != 'unknown' else ai['top']

def extract(df, name, target=None):
    if target and 'target' in df.columns: df = df[df['target']==target]
    if 'metric' in df.columns and 'value' in df.columns:
        sub = df[df['metric']==name]
        return float(sub['value'].iloc[0]) if not sub.empty else np.nan
    return float(df[name].iloc[0]) if name in df.columns else np.nan

def fmt_feats(f): return ', '.join(f) if f else 'none'
pre_steps = str(pre_manifest.get('steps', 'see manifest'))[:180]
sel_n = sel_manifest.get('n_features', 'unknown')

interp = pd.DataFrame([
    ['model_class', bi['class'], ai['class'], 'top-level estimator'],
    ['estimator_class', bi['est'], ai['est'], 'final pipeline step'],
    ['selected_hyperparameters', bi['params'], ai['params'], 'key params only'],
    ['n_features', bi['n'], ai['n'], f'selector n_features={sel_n}'],
    ['input_features', fmt_feats(bi['features']), fmt_feats(ai['features']), 'training schema'],
    ['preprocessing_steps', pre_steps, pre_steps, 'from preprocessor manifest'],
    ['top_feature', bi['top'], ai['top'], 'magnitude-based ranking'],
    ['val_metric_name', metric, metric, 'primary metric from task'],
    ['val_metric_value', extract(val_metrics, metric, targets[0]), extract(val_metrics, metric, targets[0]), 'validation split'],
    ['limitations', 'Frozen; no refit; single target; dependent on preprocessor/selector chain.', 'Frozen; no refit; risk of overfit; validation only.', 'see manifests for details'],
], columns=['aspect','baseline_model','alternative_model','notes'])

for c in ['baseline_model','alternative_model']:
    interp[c] = interp[c].apply(lambda x: f"{x:.4g}" if isinstance(x,(int,float,np.floating)) else str(x))

mlca_runtime.save_table(interp, 'interpretation.csv', caption='Frozen model interpretation')
mlca_runtime.record_fact_candidate('top_feature', top, source='interpretation.csv')
print(f"Baseline features: {bi['n']}, Alternative features: {ai['n']}")
print(f"Top feature: {top}")
print(f"Val {metric}: {extract(val_metrics, metric, targets[0])}")
