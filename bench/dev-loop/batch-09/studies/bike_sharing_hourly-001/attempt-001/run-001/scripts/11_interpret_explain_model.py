import mlca_runtime, json, pickle
import pandas as pd, numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def load_json(k):
    return json.loads(inputs[k].read_text())

def load_pkl(k):
    with open(inputs[k], 'rb') as f:
        return pickle.load(f)

proto = load_json('_protocol')
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto['group_column']
task = proto['task']
metric = task['primary_metric']

pre_m = load_json('05_preprocess_fit_preprocessing_Q23GTBD5/manifest.json')
sel_m = load_json('06_features_retain_features_R3TMNHH4/manifest.json')
base_m = load_json('07_model_train_baseline_4E4949CT/manifest.json')
alt_m = load_json('08_model_train_alternative_4C73HMKJ/manifest.json')

pre = load_pkl('05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl')
sel = load_pkl('06_features_retain_features_R3TMNHH4/selector_all.pkl')
model = load_pkl('07_model_train_baseline_4E4949CT/model.pkl')
alt = load_pkl('latest_model')

def get_feats(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if hasattr(obj, 'get_feature_names_out'):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    return None

feat_names = None
for o in [sel, pre, model, alt]:
    f = get_feats(o)
    if f:
        feat_names = f
        break
if feat_names is None:
    for m in [sel_m, pre_m, base_m, alt_m]:
        for k in ['selected_features', 'feature_names', 'features', 'input_features', 'columns']:
            if k in m and isinstance(m[k], list):
                feat_names = m[k]
                break
        if feat_names:
            break
if feat_names is None:
    tr = pd.read_parquet(inputs['train'])
    cand = [c for c in tr.columns if c not in targets + excluded + [group]]
    feat_names = []
    for c in cand:
        s = tr[c]
        if s.nunique(dropna=True) == len(s.dropna()) and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
            continue
        feat_names.append(c)

def final_est(m):
    if hasattr(m, 'named_steps'):
        return list(m.named_steps.values())[-1]
    return m

def psummary(m):
    est = final_est(m)
    p = est.get_params()
    keys = ['n_estimators', 'max_depth', 'learning_rate', 'alpha', 'C', 'penalty',
            'loss', 'min_samples_leaf', 'subsample', 'max_iter', 'l1_ratio']
    return {k: p[k] for k in keys if k in p}

top_feature = 'unavailable'
top_source = '07_model_train_baseline_4E4949CT/model.pkl'
if feat_names:
    for obj, src in [(model, top_source), (alt, '08_model_train_alternative_4C73HMKJ/alternative_model.pkl')]:
        est = final_est(obj)
        v = None
        if hasattr(est, 'feature_importances_'):
            v = est.feature_importances_
        elif hasattr(est, 'coef_'):
            v = np.abs(np.ravel(est.coef_))
        if v is not None and len(v) == len(feat_names):
            top_feature = feat_names[int(np.argmax(v))]
            top_source = src
            break
mlca_runtime.record_fact_candidate('top_feature', top_feature, source=top_source)

val = pd.read_csv(inputs['09_validate_score_validation_08S6B52Y/validation_metrics.csv'])
metric_val = None
for col in val.columns:
    if val[col].dtype == object:
        msk = val[col].astype(str).str.contains(metric, case=False, na=False)
        if msk.any():
            for c2 in val.columns:
                if pd.api.types.is_numeric_dtype(val[c2]):
                    metric_val = float(val.loc[msk, c2].iloc[0])
                    break
            if metric_val is not None:
                break
if metric_val is None:
    num = val.select_dtypes(include=[np.number]).columns
    if len(num):
        metric_val = float(val[num[0]].iloc[0])
if metric_val is not None:
    mlca_runtime.record_metric(metric, metric_val, split='val')

feat_str = ','.join(feat_names[:10]) + ('...' if feat_names and len(feat_names) > 10 else '') if feat_names else ''
rows = [
    {'section': 'model', 'key': 'type', 'value': type(model).__name__, 'numeric_value': np.nan},
    {'section': 'model', 'key': 'final_estimator', 'value': type(final_est(model)).__name__, 'numeric_value': np.nan},
    {'section': 'model', 'key': 'hyperparameters_baseline', 'value': str(psummary(model)), 'numeric_value': np.nan},
    {'section': 'model', 'key': 'hyperparameters_alt', 'value': str(psummary(alt)), 'numeric_value': np.nan},
    {'section': 'model', 'key': 'targets', 'value': ','.join(targets), 'numeric_value': np.nan},
    {'section': 'model', 'key': 'primary_metric', 'value': metric, 'numeric_value': np.nan},
    {'section': 'inputs', 'key': 'n_features', 'value': 'number of features', 'numeric_value': len(feat_names) if feat_names else 0},
    {'section': 'inputs', 'key': 'features', 'value': feat_str, 'numeric_value': np.nan},
    {'section': 'inputs', 'key': 'excluded_columns', 'value': ','.join(excluded), 'numeric_value': np.nan},
    {'section': 'inputs', 'key': 'group_column', 'value': group, 'numeric_value': np.nan},
    {'section': 'validation', 'key': metric, 'value': metric, 'numeric_value': round(metric_val, 4) if metric_val is not None else np.nan},
    {'section': 'limitations', 'key': 'frozen_no_refit', 'value': 'True', 'numeric_value': np.nan},
    {'section': 'limitations', 'key': 'validation_only', 'value': 'True', 'numeric_value': np.nan},
    {'section': 'limitations', 'key': 'no_test_exposure', 'value': 'True', 'numeric_value': np.nan},
    {'section': 'limitations', 'key': 'top_feature_source', 'value': top_source, 'numeric_value': np.nan},
]
interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, 'interpretation.csv', caption='Frozen model interpretation summary')
print('rows', len(interp), 'features', len(feat_names) if feat_names else 0)
print(metric, metric_val)
