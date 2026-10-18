import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
need = ['targets', 'excluded_columns', 'group_column', 'task']
if not all(k in proto for k in need):
    raise ValueError(f"Missing protocol keys. Available: {list(proto.keys())}")
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto['group_column']
task = proto['task']
primary_metric = task['primary_metric']
train = pd.read_parquet(inputs['train'])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 20:
        continue
    feature_cols.append(c)

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inputs['05_preprocess_fit_preprocessing_JYNCWEGN/preprocessor.pkl'])
sel = load_pkl(inputs['06_features_retain_features_HY561TZJ/selector_all.pkl'])
model = load_pkl(inputs['07_model_train_baseline_V6BDRB4H/model.pkl'])
alt = load_pkl(inputs['08_model_train_alternative_WZMYQEJM/alternative_model.pkl'])
latest = load_pkl(inputs['latest_model'])
man_b = json.loads(inputs['07_model_train_baseline_V6BDRB4H/manifest.json'].read_text())
man_a = json.loads(inputs['08_model_train_alternative_WZMYQEJM/manifest.json'].read_text())
val_met = pd.read_csv(inputs['09_validate_score_validation_HAFY4W9E/validation_metrics.csv'])

def final_est(m):
    if hasattr(m, 'named_steps') and m.named_steps:
        return list(m.named_steps.values())[-1]
    return m

rows = []
def add(section, item, value, source):
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = f"{float(value):.4g}"
    else:
        value = str(value)[:200]
    rows.append({'section': section, 'item': item, 'value': value, 'source': source})

add('task', 'primary_metric', primary_metric, '_protocol')
add('data', 'n_features', len(feature_cols), 'train.parquet')
add('data', 'n_rows_train', len(train), 'train.parquet')
add('data', 'targets', ';'.join(map(str, targets)) if targets else 'none', '_protocol')
add('data', 'excluded', ';'.join(map(str, excluded)) if excluded else 'none', '_protocol')
for name, m, src in [('baseline', model, 'model.pkl'),
                     ('alternative', alt, 'alternative_model.pkl'),
                     ('latest', latest, 'latest_model')]:
    fe = final_est(m)
    add('model', f'{name}_type', type(fe).__name__, src)
    if hasattr(fe, 'get_params'):
        params = fe.get_params()
        for k in ['n_components', 'hidden_layer_sizes', 'alpha', 'learning_rate_init',
                  'max_iter', 'solver', 'activation', 'random_state']:
            if k in params:
                add('hyperparam', f'{name}_{k}', params[k], src)
    if hasattr(m, 'named_steps'):
        add('pipeline', f'{name}_steps', '|'.join(m.named_steps.keys()), src)
if hasattr(sel, 'get_support'):
    sup = sel.get_support()
    add('selector', 'n_selected', int(np.sum(sup)), 'selector_all.pkl')

if not val_met.empty:
    for _, r in val_met.iterrows():
        split = r.get('split', 'val')
        if primary_metric in val_met.columns:
            add('validation', f'{split}_{primary_metric}', r[primary_metric], 'validation_metrics.csv')
        else:
            for c in val_met.columns:
                if pd.api.types.is_numeric_dtype(val_met[c]):
                    add('validation', f'{split}_{c}', r[c], 'validation_metrics.csv')
                    break

top_feat = None
if hasattr(sel, 'get_support'):
    sup = np.asarray(sel.get_support())
    if hasattr(sel, 'feature_names_in_') and sup.any():
        top_feat = sel.feature_names_in_[np.argmax(sup)]
if top_feat is None:
    fe = final_est(model)
    if hasattr(fe, 'feature_importances_'):
        imp = np.abs(fe.feature_importances_)
        if hasattr(fe, 'feature_names_in_'):
            top_feat = fe.feature_names_in_[np.argmax(imp)]
        elif len(feature_cols) == len(imp):
            top_feat = feature_cols[np.argmax(imp)]
    elif hasattr(fe, 'coef_'):
        coef = np.abs(np.asarray(fe.coef_))
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        if hasattr(fe, 'feature_names_in_'):
            top_feat = fe.feature_names_in_[np.argmax(coef)]
        elif len(feature_cols) == len(coef):
            top_feat = feature_cols[np.argmax(coef)]
    elif hasattr(fe, 'components_'):
        score = np.abs(fe.components_).sum(axis=0)
        if hasattr(fe, 'feature_names_in_'):
            top_feat = fe.feature_names_in_[np.argmax(score)]
        elif len(feature_cols) == len(score):
            top_feat = feature_cols[np.argmax(score)]
if top_feat is None:
    top_feat = feature_cols[0] if feature_cols else 'unknown'
add('interpretation', 'top_feature', top_feat, 'interpretation.csv')
add('limitations', 'outcome', 'no outcome; unsupervised representation', '_protocol')
add('limitations', 'group_column', 'none used' if group is None else str(group), '_protocol')
add('limitations', 'metric_note', 'reconstruction error; PCA inverse_transform, MLP predict', '_protocol')

out = pd.DataFrame(rows[:50])
mlca_runtime.save_table(out, 'interpretation.csv', caption='Frozen model interpretation')
mlca_runtime.record_fact_candidate('top_feature', top_feat, source='interpretation.csv')
print('interpretation.csv saved with', len(out), 'rows')
print('top_feature:', top_feat)
