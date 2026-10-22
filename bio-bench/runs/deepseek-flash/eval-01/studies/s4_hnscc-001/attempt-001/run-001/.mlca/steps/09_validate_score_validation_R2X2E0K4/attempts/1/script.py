import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

event_col = None
for k in ['event_column', 'event', 'status_column', 'event_col']:
    if k in protocol:
        event_col = protocol[k]
        break
if event_col is None and 'event_column' in protocol.get('task', {}):
    event_col = protocol['task']['event_column']
if event_col is None:
    raise KeyError(f"Missing event column. Top-level keys: {list(protocol.keys())}")

duration_col = targets[0]

preproc = load_pkl(inputs['05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_select_features_B4QNDNPN/selector.pkl'])
model = load_pkl(inputs['07_model_train_cox_5FF75D4Q/model.pkl'])
alt_model = load_pkl(inputs['08_model_train_alternative_6GAGDKGM/alternative_model.pkl'])

def get_raw_feature_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for name, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

feat_cols = get_raw_feature_cols(model)
if feat_cols is None:
    feat_cols = get_raw_feature_cols(alt_model)
if feat_cols is None:
    feat_cols = get_raw_feature_cols(preproc)
if feat_cols is None:
    feat_cols = get_raw_feature_cols(selector)
if feat_cols is None:
    drop = set(targets + excluded + ([group_col] if group_col else []))
    feat_cols = [c for c in train.columns if c not in drop]
    for c in feat_cols[:]:
        s = train[c]
        if s.dtype == object or str(s.dtype) == 'string':
            if s.nunique() == len(s):
                feat_cols.remove(c)
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique() == len(s) and s.is_monotonic_increasing:
                feat_cols.remove(c)

def transform_and_predict(model, X):
    if isinstance(model, Pipeline):
        Xt = X
        for name, step in model.steps[:-1]:
            Xt = step.transform(Xt)
        final = model.steps[-1][1]
        if hasattr(final, 'predict_partial_hazard'):
            risk = final.predict_partial_hazard(Xt)
        else:
            raise ValueError("Final step lacks predict_partial_hazard")
        return risk, Xt, final
    elif isinstance(model, CoxPHFitter):
        risk = model.predict_partial_hazard(X)
        return risk, X, model
    else:
        Xt = preproc.transform(X)
        Xt = selector.transform(Xt)
        risk = model.predict_partial_hazard(Xt)
        return risk, Xt, model

def to_binary_event(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    mapping = {'Dead':1, 'Alive':0, 'DECEASED':1, 'LIVING':0, '1':1, '0':0, 'True':1, 'False':0}
    s2 = s.map(mapping)
    if s2.isna().any():
        codes, uniques = pd.factorize(s)
        if len(uniques) == 2:
            return pd.Series(codes, index=s.index)
        raise ValueError(f"Cannot convert event column {event_col} to binary")
    return s2.astype(int)

val_event = to_binary_event(val[event_col])
val_time = pd.to_numeric(val[duration_col], errors='coerce')

models = {'baseline': model, 'alternative': alt_model}
results = []
c_indices = {}
for name, m in models.items():
    risk, Xt, cph = transform_and_predict(m, val[feat_cols])
    ci = concordance_index(val_time, -risk, val_event)
    c_indices[name] = ci
    results.append({'model': name, 'metric': primary_metric, 'value': ci})
    mlca_runtime.record_metric(primary_metric, ci, 'val')

ph_rows = []
try:
    _, Xt_base, cph_base = transform_and_predict(model, val[feat_cols])
    df_ph = Xt_base.copy()
    df_ph[duration_col] = val_time.values
    df_ph[event_col] = val_event.values
    ph = proportional_hazard_test(cph_base, df_ph, time_transform='rank')
    pvals = ph.summary['p']
    n_viol = int((pvals < 0.05).sum())
    ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_n_p_lt_0.05', 'value': n_viol})
    ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_min_p', 'value': float(pvals.min())})
    for cov, p in pvals.sort_values().head(40).items():
        ph_rows.append({'model': 'baseline', 'metric': f'schoenfeld_p_{cov}', 'value': float(p)})
except Exception:
    pass

df_metrics = pd.DataFrame(results + ph_rows)
df_metrics['value'] = df_metrics['value'].apply(
    lambda x: float(f'{x:.4g}') if isinstance(x, (int, float, np.floating)) else x
)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics')

best_name = max(c_indices, key=c_indices.get)
best_ci = c_indices[best_name]
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_c_index', best_ci, source='validation_metrics.csv')

print(f"baseline c-index: {c_indices.get('baseline', np.nan):.4f}")
print(f"alternative c-index: {c_indices.get('alternative', np.nan):.4f}")
print(f"best model: {best_name} ({best_ci:.4f})")
