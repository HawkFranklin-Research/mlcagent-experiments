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
val = pd.read_parquet(inputs['val'])
train = pd.read_parquet(inputs['train'])

task = protocol['task']
primary_metric = task['primary_metric']
targets = list(protocol['targets'])
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
    raise KeyError(f"Missing event column. Keys: {list(protocol.keys())}")

duration_col = targets[0]

preproc = load_pkl(inputs['05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_select_features_B4QNDNPN/selector.pkl'])
model = load_pkl(inputs['07_model_train_cox_5FF75D4Q/model.pkl'])
alt_model = load_pkl(inputs['08_model_train_alternative_6GAGDKGM/alternative_model.pkl'])

def feat_inputs(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    steps = getattr(obj, 'named_steps', None)
    if steps:
        for s in steps.values():
            if hasattr(s, 'feature_names_in_'):
                return list(s.feature_names_in_)
    steps = getattr(obj, 'steps', None)
    if steps:
        for _, s in steps:
            if hasattr(s, 'feature_names_in_'):
                return list(s.feature_names_in_)
    return None

raw_cols = feat_inputs(preproc) or feat_inputs(selector) or feat_inputs(model)
if raw_cols is None:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    raw_cols = [c for c in val.columns if c not in drop]

def to_binary_event(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    mp = {'Dead': 1, 'DECEASED': 1, '1': 1, 'True': 1,
          'Alive': 0, 'LIVING': 0, '0': 0, 'False': 0}
    s2 = s.map(mp)
    if s2.isna().any():
        codes, uniques = pd.factorize(s)
        if len(uniques) == 2:
            return pd.Series(codes, index=s.index)
        raise ValueError(f"Cannot convert event column {event_col}")
    return s2.astype(int)

val_event = to_binary_event(val[event_col])
val_time = pd.to_numeric(val[duration_col], errors='coerce')
Xraw = val[[c for c in raw_cols if c in val.columns]].copy()

def transform_features(X):
    sel_cols = feat_inputs(selector)
    if sel_cols is not None and all(c in X.columns for c in sel_cols):
        return selector.transform(X[sel_cols])
    try:
        return selector.transform(X)
    except Exception:
        return selector.transform(preproc.transform(X))

def find_cox(m):
    if isinstance(m, CoxPHFitter):
        return m
    for a in ('coxph_', 'cox', 'estimator', 'final_estimator', 'fitter', '_fitter'):
        v = getattr(m, a, None)
        if isinstance(v, CoxPHFitter):
            return v
    if isinstance(m, Pipeline):
        last = m.steps[-1][1]
        if isinstance(last, CoxPHFitter):
            return last
    return None

def predict_risk(m, X):
    if isinstance(m, Pipeline):
        Xt = X
        for _, step in m.steps[:-1]:
            Xt = step.transform(Xt)
        final = m.steps[-1][1]
        return np.asarray(final.predict_partial_hazard(Xt)).ravel(), Xt, final
    try:
        return np.asarray(m.predict_partial_hazard(X)).ravel(), X, m
    except Exception:
        pass
    Xt = transform_features(X)
    return np.asarray(m.predict_partial_hazard(Xt)).ravel(), Xt, m

results, c_indices, fitted = [], {}, {}
for name, m in [('baseline', model), ('alternative', alt_model)]:
    risk, Xt, cph = predict_risk(m, Xraw)
    ci = float(concordance_index(val_time, -risk, val_event))
    c_indices[name] = ci
    fitted[name] = (Xt, cph)
    results.append({'model': name, 'metric': primary_metric, 'value': round(ci, 4)})
    mlca_runtime.record_metric(primary_metric, ci, 'val')

ph_rows = []
Xt_b, cph_b = fitted['baseline']
cox = find_cox(cph_b)
if cox is not None:
    if isinstance(Xt_b, pd.DataFrame):
        df_ph = Xt_b.copy()
    else:
        names = list(getattr(cox, 'params_', pd.Series(dtype=float)).index)
        df_ph = pd.DataFrame(np.asarray(Xt_b), columns=names, index=val.index)
    df_ph[duration_col] = val_time.values
    df_ph[event_col] = val_event.values
    try:
        ph = proportional_hazard_test(cox, df_ph, time_transform='rank')
        pv = ph.summary['p']
        ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_n_p_lt_0.05',
                        'value': int((pv < 0.05).sum())})
        ph_rows.append({'model': 'baseline', 'metric': 'schoenfeld_min_p',
                        'value': round(float(pv.min()), 4)})
        for cov, p in pv.sort_values().head(40).items():
            ph_rows.append({'model': 'baseline', 'metric': f'schoenfeld_p_{cov}',
                            'value': round(float(p), 4)})
    except Exception as e:
        print(f"PH test failed: {type(e).__name__}")

df_metrics = pd.DataFrame(results + ph_rows)
mlca_runtime.save_table(df_metrics, 'validation_metrics.csv', caption='Validation metrics')

best = max(c_indices, key=c_indices.get)
mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_c_index', round(c_indices[best], 4),
                                   source='validation_metrics.csv')

print(f"baseline c-index={c_indices['baseline']:.4f}")
print(f"alternative c-index={c_indices['alternative']:.4f}")
print(f"best={best} ({c_indices[best]:.4f})")
