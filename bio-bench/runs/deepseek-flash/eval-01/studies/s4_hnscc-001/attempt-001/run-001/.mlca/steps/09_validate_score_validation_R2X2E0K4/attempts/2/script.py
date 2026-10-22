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

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column', None)

def getk(d, path):
    v = d
    for k in path:
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v

event_col = None
for path in [('event_column',), ('event_col',), ('status_column',), ('event',),
             ('columns', 'event_column'), ('columns', 'event'), ('columns', 'status'),
             ('task', 'event_column'), ('task', 'event'), ('task', 'status_column'),
             ('task', 'columns', 'event_column'), ('outcomes', 'event_column'),
             ('outcomes', 'event'), ('outcome', 'event_column')]:
    v = getk(protocol, path)
    if isinstance(v, str) and v:
        event_col = v
        break
if event_col is None:
    raise KeyError(f"Missing event column. Top-level keys: {list(protocol.keys())}; "
                   f"task keys: {list(protocol.get('task', {}).keys())}")

duration_col = targets[0]

preproc = load_pkl(inputs['05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_select_features_B4QNDNPN/selector.pkl'])
model = load_pkl(inputs['07_model_train_cox_5FF75D4Q/model.pkl'])
alt_model = load_pkl(inputs['08_model_train_alternative_6GAGDKGM/alternative_model.pkl'])

def get_raw_feature_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

feat_cols = (get_raw_feature_cols(model) or get_raw_feature_cols(alt_model)
             or get_raw_feature_cols(preproc) or get_raw_feature_cols(selector))
if feat_cols is None:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feat_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if (s.dtype == object or str(s.dtype) == 'string') and s.nunique() == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and s.is_monotonic_increasing:
            continue
        feat_cols.append(c)

def to_binary_event(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    mapping = {'Dead':1,'DECEASED':1,'DECEASED ':1,'1':1,'True':1,
               'Alive':0,'LIVING':0,'LIVING ':0,'0':0,'False':0}
    s2 = s.map(mapping)
    if s2.isna().any():
        codes, uniques = pd.factorize(s)
        if len(uniques) == 2:
            return pd.Series(codes, index=s.index)
        raise ValueError(f"Cannot convert event column {event_col}")
    return s2.astype(int)

def predict_risk(m, Xraw):
    if isinstance(m, Pipeline):
        Xt = Xraw
        for _, step in m.steps[:-1]:
            Xt = step.transform(Xt)
        final = m.steps[-1][1]
        return final.predict_partial_hazard(Xt), Xt, final
    if isinstance(m, CoxPHFitter):
        try:
            return m.predict_partial_hazard(Xraw), Xraw, m
        except (KeyError, ValueError):
            Xt = selector.transform(preproc.transform(Xraw))
            return m.predict_partial_hazard(Xt), Xt, m
    Xt = selector.transform(preproc.transform(Xraw))
    return m.predict_partial_hazard(Xt), Xt, m

val_event = to_binary_event(val[event_col])
val_time = pd.to_numeric(val[duration_col], errors='coerce')
Xval_raw = val[feat_cols]

results, c_indices, fitted = [], {}, {}
for name, m in [('baseline', model), ('alternative', alt_model)]:
    risk, Xt, cph = predict_risk(m, Xval_raw)
    ci = float(concordance_index(val_time, -risk, val_event))
    c_indices[name] = ci
    fitted[name] = (Xt, cph)
    results.append({'model': name, 'metric': primary_metric, 'value': float(f'{ci:.4g}')})
    mlca_runtime.record_metric(primary_metric, ci, 'val')

ph_rows = []
try:
    Xt, cph = fitted['baseline']
    df_ph = Xt.copy()
    df_ph[duration_col] = val_time.values
    df_ph[event_col] = val_event.values
    ph = proportional_hazard_test(cph, df_ph, time_transform='rank')
    pvals = ph.summary['p']
    ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_n_p_lt_0.05',
                    'value': int((pvals < 0.05).sum())})
    ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_min_p',
                    'value': float(f'{pvals.min():.4g}')})
    for cov, p in pvals.sort_values().head(40).items():
        ph_rows.append({'model': 'baseline', 'metric': f'schoenfeld_p_{cov}',
                        'value': float(f'{float(p):.4g}')})
except Exception as e:
    print(f"PH test skipped: {type(e).__name__}")

df_metrics = pd.DataFrame(results + ph_rows)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics')

best_name = max(c_indices, key=c_indices.get)
best_ci = c_indices[best_name]
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_c_index', float(f'{best_ci:.4g}'),
                                   source='validation_metrics.csv')

print(f"baseline c-index: {c_indices['baseline']:.4f}")
print(f"alternative c-index: {c_indices['alternative']:.4f}")
print(f"best: {best_name} ({best_ci:.4f})")
