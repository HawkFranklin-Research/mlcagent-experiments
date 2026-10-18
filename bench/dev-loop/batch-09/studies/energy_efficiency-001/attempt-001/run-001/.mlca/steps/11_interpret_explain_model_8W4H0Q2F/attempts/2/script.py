import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs['_protocol'].read_text())
targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns', []))
group = proto.get('group_column')
task = proto.get('task', {})
metric = task.get('primary_metric', 'rmse')

def load_json(k):
    return json.loads(inputs[k].read_text())

def load_pkl(k):
    with open(inputs[k], 'rb') as f:
        return pickle.load(f)

val_df = pd.read_csv(inputs['09_validate_score_validation_N590X4VJ/validation_metrics.csv'])
baseline = load_pkl('07_model_train_baseline_Q7Z8BHZ9/model.pkl')
alt = load_pkl('08_model_train_alternative_5G4S54TV/alternative_model.pkl')
sel = load_pkl('06_features_retain_features_JTARV5X3/selector_all.pkl')
pre = load_pkl('05_preprocess_fit_preprocessing_0CQ66EDA/preprocessor.pkl')

def final_est(m):
    return list(m.named_steps.values())[-1] if hasattr(m, 'named_steps') else m

def params_dict(m):
    try:
        p = final_est(m).get_params(deep=False)
    except Exception:
        p = {}
    r = {}
    for k, v in p.items():
        if isinstance(v, (int, float, bool, str, type(None))):
            r[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 8 and all(isinstance(x, (int, float, bool, str, type(None))) for x in v):
            r[k] = list(v)
    return r

def feat_names(model, sel, pre):
    for obj in (model, sel, pre):
        if hasattr(obj, 'named_steps'):
            try:
                return list(obj[:-1].get_feature_names_out())
            except Exception:
                pass
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    return None

def imps(est):
    if hasattr(est, 'estimators_'):
        arr = [imps(e) for e in est.estimators_]
        arr = [a for a in arr if a is not None]
        if arr:
            return np.mean(np.vstack(arr), axis=0)
    if hasattr(est, 'feature_importances_'):
        return np.asarray(est.feature_importances_, dtype=float)
    if hasattr(est, 'coef_'):
        c = np.asarray(est.coef_, dtype=float)
        return np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c).ravel()
    return None

names = feat_names(baseline, sel, pre)
if not names:
    names = feat_names(alt, sel, pre)
imp = imps(final_est(baseline))
if imp is None:
    imp = imps(final_est(alt))
top_feature = str(names[int(np.argmax(imp))]) if (names and imp is not None and len(imp) == len(names)) else 'unknown'

def sc(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

rows = [
    ('Model', 'baseline_estimator', type(final_est(baseline)).__name__, '07_model_train_baseline_Q7Z8BHZ9/model.pkl'),
    ('Model', 'alternative_estimator', type(final_est(alt)).__name__, '08_model_train_alternative_5G4S54TV/alternative_model.pkl'),
    ('Inputs', 'targets', ','.join(targets), '_protocol'),
    ('Inputs', 'excluded_columns', ','.join(excluded) if excluded else '(none)', '_protocol'),
    ('Inputs', 'group_column', str(group) if group else '(none)', '_protocol'),
    ('Inputs', 'primary_metric', str(metric), '_protocol'),
    ('Inputs', 'model_input_width', str(len(names) if names else 'unknown'), '06_features_retain_features_JTARV5X3/selector_all.pkl'),
]
for k, v in params_dict(baseline).items():
    rows.append(('Hyperparameters', 'baseline.' + k, sc(v), '07_model_train_baseline_Q7Z8BHZ9/model.pkl'))
for k, v in params_dict(alt).items():
    rows.append(('Hyperparameters', 'alternative.' + k, sc(v), '08_model_train_alternative_5G4S54TV/alternative_model.pkl'))

if 'metric' in val_df.columns and 'value' in val_df.columns:
    for _, r in val_df.iterrows():
        t = r.get('target', '') if 'target' in val_df.columns else ''
        label = f"{r['metric']}[{t}]" if t else str(r['metric'])
        try:
            v = float(r['value'])
        except Exception:
            continue
        rows.append(('Validation', label, sc(v), '09_validate_score_validation_N590X4VJ/validation_metrics.csv'))
else:
    for c in val_df.columns:
        for _, r in val_df.iterrows():
            try:
                v = float(r[c])
            except Exception:
                continue
            rows.append(('Validation', str(c), sc(v), '09_validate_score_validation_N590X4VJ/validation_metrics.csv'))

rows.append(('Facts', 'top_feature', top_feature, '07_model_train_baseline_Q7Z8BHZ9/model.pkl'))
for i, l in enumerate([
    'Only 768 simulated configurations; no repeated units or external cohorts.',
    'Ensemble baselines do not extrapolate beyond the observed range of the eight design factors.',
    'Two correlated targets (Y1, Y2) modeled jointly; per-target RMSE reported alongside the mean.',
    'No temporal or identifier columns; generalization to other building datasets is not established.',
], 1):
    rows.append(('Limitations', f'L{i}', l, '09_validate_score_validation_N590X4VJ/manifest.json'))

df = pd.DataFrame(rows, columns=['section', 'item', 'value', 'source_file'])
if len(df) > 50:
    keep = df[df['section'] != 'Hyperparameters']
    extra = df[df['section'] == 'Hyperparameters'].head(max(0, 50 - len(keep)))
    df = pd.concat([keep, extra], ignore_index=True).iloc[:50]
df['value'] = df['value'].astype(str)

mlca_runtime.save_table(df, 'interpretation.csv', caption='Frozen multioutput regression model interpretation')
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
print(f"rows={len(df)} top_feature={top_feature} width={len(names) if names else 'unknown'}")
