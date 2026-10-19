import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in proto:
        raise KeyError(f"Missing {k}. Available: {list(proto.keys())}")
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column')
task = proto['task']

train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])

excl = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = []
for c in train.columns:
    if c in excl:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and any(k in c.lower() for k in ['id','code','key','index']):
        continue
    feat_cols.append(c)

def load_pkl(key):
    return pickle.loads(paths[key].read_bytes())

pre = load_pkl('05_preprocess_fit_preprocessing_5Z83A35F/preprocessor.pkl')
sel = load_pkl('06_features_retain_features_Z9TAK0X7/selector_all.pkl')
model = load_pkl('07_model_train_baseline_AHTSSAGQ/model.pkl')
latest = load_pkl('latest_model')
alt = load_pkl('08_model_train_alternative_AC8TT8JH/alternative_model.pkl')
val_metrics = pd.read_csv(paths['09_validate_score_validation_CWBDSN2A/validation_metrics.csv'])

def get_final_est(m):
    while hasattr(m, 'steps'):
        m = m.steps[-1][1]
    if hasattr(m, 'best_estimator_'):
        m = m.best_estimator_
        while hasattr(m, 'steps'):
            m = m.steps[-1][1]
    return m

est = get_final_est(model)
imp = None
if hasattr(est, 'feature_importances_'):
    imp = np.asarray(est.feature_importances_)
elif hasattr(est, 'coef_'):
    coef = est.coef_
    if isinstance(coef, list):
        imp = np.mean([np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c) for c in coef], axis=0)
    else:
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

top_feat = None
if imp is not None:
    names = None
    if hasattr(est, 'feature_names_in_'):
        names = list(est.feature_names_in_)
    elif hasattr(model, 'feature_names_in_'):
        names = list(model.feature_names_in_)
    if names is None and hasattr(pre, 'get_feature_names_out'):
        try: names = list(pre.get_feature_names_out())
        except Exception: pass
    if names is None and hasattr(sel, 'get_feature_names_out'):
        try: names = list(sel.get_feature_names_out())
        except Exception: pass
    if names is None and len(imp) == len(feat_cols):
        names = feat_cols
    if names is not None and len(names) == len(imp):
        top_feat = str(names[int(np.argmax(imp))])

rows = []
rows.append({'section': 'model', 'item': 'estimator', 'value': type(est).__name__, 'detail': str(est)[:200]})
params = est.get_params()
for k in ['n_estimators', 'max_depth', 'learning_rate', 'C', 'solver', 'max_iter', 'n_neighbors', 'min_samples_split']:
    if k in params:
        rows.append({'section': 'hyperparameters', 'item': k, 'value': str(params[k]), 'detail': 'selected'})
rows.append({'section': 'inputs', 'item': 'n_features', 'value': str(len(feat_cols)), 'detail': ','.join(feat_cols[:10]) + ('...' if len(feat_cols) > 10 else '')})
rows.append({'section': 'inputs', 'item': 'targets', 'value': ','.join(targets), 'detail': 'multilabel'})
if hasattr(pre, 'transformers_'):
    tr_names = [str(t[0]) for t in pre.transformers_]
    rows.append({'section': 'inputs', 'item': 'preprocessor_transformers', 'value': str(len(tr_names)), 'detail': ','.join(tr_names)})
if hasattr(sel, 'get_support'):
    try:
        sup = sel.get_support()
        rows.append({'section': 'inputs', 'item': 'selected_features', 'value': str(int(sup.sum())), 'detail': f'from {len(sup)}'})
    except Exception:
        pass
rows.append({'section': 'validation', 'item': 'metrics_file', 'value': '09_validate_score_validation_CWBDSN2A/validation_metrics.csv', 'detail': f'{val_metrics.shape[0]} rows'})
pm = task.get('primary_metric')
if pm and 'metric' in val_metrics.columns and 'value' in val_metrics.columns:
    mrow = val_metrics[val_metrics['metric'] == pm]
    if len(mrow):
        rows.append({'section': 'validation', 'item': str(pm), 'value': f"{float(mrow['value'].iloc[0]):.4g}", 'detail': 'primary metric'})
for _, r in val_metrics.iterrows():
    if r.get('metric') != pm:
        rows.append({'section': 'validation', 'item': str(r.get('metric')), 'value': str(r.get('value')), 'detail': ''})
rows.append({'section': 'limitations', 'item': 'single_center', 'value': 'Cleveland clinic', 'detail': 'no external validation'})
rows.append({'section': 'limitations', 'item': 'multilabel', 'value': 'Two targets', 'detail': 'num and heart_disease_binary are recodings; no independent test'})
rows.append({'section': 'limitations', 'item': 'frozen_model', 'value': 'No refit', 'detail': 'Interpretation only; no new fitting'})

interp_df = pd.DataFrame(rows).head(50)[['section', 'item', 'value', 'detail']]
for col in interp_df.columns:
    interp_df[col] = interp_df[col].astype(str)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Frozen model interpretation")

if top_feat and top_feat != "unknown":
    mlca_runtime.record_fact_candidate("top_feature", top_feat, source="interpretation.csv")

print(f"features={len(feat_cols)}")
print(f"top_feature={top_feat}")
if pm and 'metric' in val_metrics.columns and 'value' in val_metrics.columns:
    mrow = val_metrics[val_metrics['metric'] == pm]
    if len(mrow):
        print(f"{pm}={float(mrow['value'].iloc[0]):.4g}")
