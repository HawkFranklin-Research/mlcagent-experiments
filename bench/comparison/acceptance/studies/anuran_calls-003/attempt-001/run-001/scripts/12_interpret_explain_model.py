import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets, task = proto['targets'], proto['task']
primary = task['primary_metric']

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

prep = load(inp['06_preprocess_fit_preprocessing_D9KP7ADR/preprocessor.pkl'])
sel  = load(inp['07_features_retain_features_11DAHWEE/selector_all.pkl'])
base = load(inp['08_model_train_baseline_SJFKVA7H/model.pkl'])
alt  = load(inp['09_model_train_alternative_GFWY19ZA/alternative_model.pkl'])

def unwrap(m):
    if isinstance(m, Pipeline):
        return m[-1], m
    if isinstance(m, dict):
        for k in ('model', 'estimator', 'pipeline'):
            if k in m:
                v = m[k]
                return (v[-1], v) if isinstance(v, Pipeline) else (v, None)
    return m, None

be, bp = unwrap(base); ae, ap = unwrap(alt)

def hyper(est):
    try:
        p = est.get_params(deep=False)
    except Exception:
        return {}
    return {k: v for k, v in p.items() if isinstance(v, (int, float, str, bool))}

def feat_names(*objs):
    for o in objs:
        if o is None:
            continue
        if hasattr(o, 'get_feature_names_out'):
            try: return list(o.get_feature_names_out())
            except Exception: pass
        if hasattr(o, 'named_steps'):
            st = list(o.named_steps.values())[-1]
            if hasattr(st, 'get_feature_names_out'):
                try: return list(st.get_feature_names_out())
                except Exception: pass
    return None

fn = feat_names(sel, prep, bp, ap)
imp = None
for o in (sel, prep, be, bp, ae, ap):
    if o is None:
        continue
    if hasattr(o, 'coef_'):
        try:
            c = o.coef_
            imp = [abs(x) for x in (c.mean(axis=0) if getattr(c, 'ndim', 1) > 1 else c)]
            break
        except Exception: pass
    if hasattr(o, 'feature_importances_'):
        imp = list(o.feature_importances_); break

top_feature = None
if imp is not None and fn is not None and len(imp) == len(fn):
    top_feature = sorted(zip(fn, imp), key=lambda x: -x[1])[0][0]

val = pd.read_csv(inp['10_validate_score_validation_7QPESJ10/validation_metrics.csv'])
def pick_metric(df, name):
    for c in df.columns:
        if str(c).lower() == name.lower():
            try: return float(df.iloc[0][c])
            except Exception: pass
    for c in df.columns:
        if str(c).lower() in ('metric', 'name', 'key'):
            for _, row in df.iterrows():
                if str(row[c]).lower() == name.lower():
                    for vc in df.columns:
                        try: return float(row[vc])
                        except Exception: continue
    return None
vm = pick_metric(val, primary)
if vm is None:
    for c in val.columns:
        s = pd.to_numeric(val[c], errors='coerce').dropna()
        if len(s):
            vm = float(s.iloc[0]); break

rows = []
def add(s, i, v, src):
    rows.append({'section': s, 'item': i, 'value': str(v), 'source': src})

add('protocol', 'targets', ','.join(targets), '_protocol')
add('protocol', 'primary_metric', primary, '_protocol')
add('protocol', 'task_type', task.get('type', ''), '_protocol')
add('baseline', 'estimator', type(be).__name__, '08_model_train_baseline_SJFKVA7H/model.pkl')
for k, v in list(hyper(be).items())[:10]:
    add('baseline', 'hp:' + k, v, '08_model_train_baseline_SJFKVA7H/model.pkl')
add('alternative', 'estimator', type(ae).__name__, '09_model_train_alternative_GFWY19ZA/alternative_model.pkl')
for k, v in list(hyper(ae).items())[:10]:
    add('alternative', 'hp:' + k, v, '09_model_train_alternative_GFWY19ZA/alternative_model.pkl')
add('features', 'preprocessor', type(prep).__name__, '06_preprocess_fit_preprocessing_D9KP7ADR/preprocessor.pkl')
add('features', 'selector', type(sel).__name__, '07_features_retain_features_11DAHWEE/selector_all.pkl')
add('features', 'n_selected', len(fn) if fn else 'unknown', '07_features_retain_features_11DAHWEE/selector_all.pkl')
if top_feature:
    add('features', 'top_feature', top_feature, '07_features_retain_features_11DAHWEE/selector_all.pkl')
if vm is not None:
    add('validation', primary, round(vm, 4), '10_validate_score_validation_7QPESJ10/validation_metrics.csv')
add('limitations', 'single_split', 'one hold-out validation; no cross-validation', '10_validate_score_validation_7QPESJ10/validation_metrics.csv')
add('limitations', 'nested_labels', 'Family>Genus>Species; errors correlate across targets', '_protocol')
add('limitations', 'no_refit', 'frozen interpretation only; model not refit', '08_model_train_baseline_SJFKVA7H/model.pkl')

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, 'interpretation.csv', caption='Frozen model interpretation')

if vm is not None:
    mlca_runtime.record_metric(primary, float(vm), 'val')
if top_feature:
    mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

for line in [f"rows={len(interp)}", f"primary={primary}", f"val={vm}", f"top_feature={top_feature}"]:
    print(line)
