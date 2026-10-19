import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
P = mlca_runtime.inputs()

def jload(k):
    return json.loads(P[k].read_text())

def pload(k):
    with open(P[k], 'rb') as f:
        return pickle.load(f)

def need(d, k, top=None):
    if k not in d:
        raise KeyError(f"Missing {k!r}; available: {list(top if top is not None else d)}")
    return d[k]

proto = jload('_protocol')
task = need(proto, 'task')
primary_metric = need(task, 'primary_metric', top=proto)
targets = need(proto, 'targets')
excluded = need(proto, 'excluded_columns')
group = need(proto, 'group_column')
train = pd.read_parquet(P['train'])
val = pd.read_parquet(P['val'])

def keylike(s):
    if not pd.api.types.is_numeric_dtype(s):
        return True
    return pd.api.types.is_integer_dtype(s) and len(s) > 0 and s.nunique(dropna=True) == len(s)

feature_cols = [c for c in train.columns
                if c not in set(targets) | set(excluded) | {group} and not keylike(train[c])]

model_paths = {
    'baseline': '07_model_train_baseline_KG9SJ6DB/model.pkl',
    'alternative': '08_model_train_alternative_WEEPWNFD/alternative_model.pkl',
    'selected': 'latest_model',
}
models = {k: pload(v) for k, v in model_paths.items()}
sel = models['selected']

def final_est(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

def prim(v):
    return isinstance(v, (str, int, float, bool, type(None), np.integer, np.floating))

def hyper(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if prim(v) and any(t in kl for t in ('param', 'hyper', 'n_components', 'alpha', 'hidden', 'max_iter', 'seed', 'selected', 'best')):
                out[prefix + str(k)] = v
            elif isinstance(v, dict):
                out.update(hyper(v, prefix + str(k) + '.'))
    return out

def params(est):
    try:
        p = est.get_params(deep=False)
    except Exception:
        return {}
    return {k: v for k, v in p.items() if prim(v)}

def names_for(model, est, n):
    for obj in (est, model):
        if hasattr(obj, 'feature_names_in_'):
            names = list(obj.feature_names_in_)
            if len(names) == n:
                return names
        if hasattr(obj, 'get_feature_names_out'):
            try:
                names = list(obj.get_feature_names_out())
                if len(names) == n:
                    return names
            except Exception:
                pass
    if isinstance(model, Pipeline):
        for _, step in reversed(model.steps[:-1]):
            if hasattr(step, 'get_feature_names_out'):
                try:
                    names = list(step.get_feature_names_out())
                    if len(names) == n:
                        return names
                except Exception:
                    pass
    if len(feature_cols) == n:
        return feature_cols
    return None

def imp_vec(est):
    if hasattr(est, 'components_'):
        c = np.asarray(est.components_)
        return np.abs(c[0]) if c.ndim == 2 else np.abs(c)
    if hasattr(est, 'feature_importances_'):
        return np.abs(np.asarray(est.feature_importances_))
    if hasattr(est, 'coefs_') and getattr(est, 'coefs_', None):
        return np.sum(np.abs(np.asarray(est.coefs_[0])), axis=1)
    if hasattr(est, 'coef_'):
        c = np.asarray(est.coef_)
        return np.abs(c) if c.ndim == 1 else np.sum(np.abs(c), axis=0)
    return None

est = final_est(sel)
vec = imp_vec(est)
top_name = None
if vec is not None and len(vec) > 0:
    i = int(np.argmax(vec))
    names = names_for(sel, est, len(vec))
    if names is not None:
        top_name = str(names[i])
if top_name is None and feature_cols:
    v = train[feature_cols].var(numeric_only=True).dropna()
    if len(v):
        top_name = str(v.idxmax())

rows = []
def add(section, item, value, source):
    if len(rows) >= 50:
        return
    if isinstance(value, (float, np.floating)) and not pd.isna(value):
        value = float(f"{value:.4g}")
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    if value is None or value == '':
        value = np.nan
    rows.append((section, item, value, source))

add('protocol', 'primary_metric', primary_metric, '_protocol')
add('protocol', 'targets', json.dumps(targets), '_protocol')
add('protocol', 'excluded_columns', json.dumps(excluded), '_protocol')
add('protocol', 'group_column', str(group), '_protocol')
add('data', 'train_rows', len(train), 'train.parquet')
add('data', 'val_rows', len(val), 'val.parquet')
add('data', 'feature_count', len(feature_cols), 'train.parquet')
const = [c for c in feature_cols if train[c].nunique(dropna=False) <= 1]
add('data', 'constant_features', ', '.join(map(str, const[:12])) + (f' (+{len(const)-12})' if len(const) > 12 else ''), 'train.parquet')
add('model', 'selected_source', 'latest_model', 'latest_model')
add('model', 'selected_class', type(est).__name__, 'latest_model')

man_key = '08_model_train_alternative_WEEPWNFD/manifest.json'
man = jload(man_key)
for k, v in list(hyper(man).items())[:10]:
    add('hyperparameters', k, v, man_key)
for k, v in list(params(est).items())[:10]:
    add('model_params', k, v, 'latest_model')

vm_key = '09_validate_score_validation_QC4M92J8/validation_metrics.csv'
if P[vm_key].exists():
    vm = pd.read_csv(P[vm_key])
else:
    vm = pd.read_parquet(P['09_validate_score_validation_QC4M92J8/validation_metrics.parquet'])
if {'metric', 'value'}.issubset(vm.columns):
    for _, r in vm.head(12).iterrows():
        label = str(r['metric'])
        if 'model' in vm.columns:
            label = f"{r['model']}:{label}"
        add('validation', label, r['value'], vm_key)
else:
    for c in list(vm.columns)[:10]:
        if pd.api.types.is_numeric_dtype(vm[c]):
            add('validation', c, vm[c].iloc[0], vm_key)

add('limitation', 'no_supervised_outcome', 'Reconstruction error only; no outcome column exists.', '_protocol')
add('limitation', 'independent_rows', 'No subject, time, event or repeated-measure identifiers are provided.', '_protocol')
add('limitation', 'validation_only', 'Selection evidence is validation-only; no test exposure or post-test fitting.', vm_key)
add('limitation', 'degenerate_pixels', 'Constant pixel features are noted as degenerate and are not interpreted as signal.', 'train.parquet')
add('inspection', 'top_feature', top_name if top_name else 'unavailable', 'latest_model')

out = pd.DataFrame(rows, columns=['section', 'item', 'value', 'source'])
mlca_runtime.save_table(out, 'interpretation.csv', 'Frozen representation model interpretation')

mlca_runtime.record_fact_candidate('top_feature', top_name if top_name else 'unavailable', source='interpretation.csv')

print(f"train/val rows: {len(train)}/{len(val)}; features: {len(feature_cols)}")
print(f"selected model: {type(est).__name__}")
print(f"top feature: {top_name if top_name else 'unavailable'}")
print(f"wrote interpretation.csv ({len(out)} rows)")
