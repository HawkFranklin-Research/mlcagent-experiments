import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []) or [])
group_col = protocol.get('group_column')
task = protocol['task']
metric = task['primary_metric']
ttype = task.get('type', 'binary')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

non_features = set(targets) | set(excluded)
if group_col:
    non_features.add(group_col)

feature_cols = []
for c in train.columns:
    if c in non_features:
        continue
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and s.nunique() == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        u = np.sort(s.unique())
        if len(u) > 1 and (u[-1] - u[0]) == len(u) - 1:
            continue
    feature_cols.append(c)
print(f"n_features={len(feature_cols)}")

def _load(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

cand = {}
for name, key in [
    ('model', '07_model_train_logistic_X3NX87MM/model.pkl'),
    ('latest_model', 'latest_model'),
    ('preprocessor', '05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl'),
    ('selector', '06_features_select_features_HGTRMBKM/selector.pkl'),
]:
    try:
        cand[name] = _load(key)
    except Exception:
        cand[name] = None

X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()
X_probe = X_val.head(3)
print(f"val_rows={len(X_val)}")

def works(obj):
    if obj is None or not hasattr(obj, 'predict_proba'):
        return False
    try:
        obj.predict_proba(X_probe)
        return True
    except Exception:
        return False

pipeline = None
for name in ['model', 'latest_model']:
    if works(cand.get(name)):
        pipeline = cand[name]
        print(f"pipeline={name}")
        break

if pipeline is None:
    pre, sel, est = cand.get('preprocessor'), cand.get('selector'), cand.get('model')
    embedded = isinstance(est, Pipeline) and any(s is pre or s is sel for _, s in est.steps)
    if embedded:
        pipeline = est
    else:
        steps = [(n, o) for n, o in [('pre', pre), ('sel', sel)] if o is not None]
        steps.append(('est', est))
        pipeline = Pipeline(steps)
    pipeline.predict_proba(X_probe)
    print("pipeline=assembled")

try:
    fni = list(pipeline.feature_names_in_)
    if set(fni) == set(feature_cols):
        feature_cols = fni
        X_val = val[feature_cols].copy()
except AttributeError:
    pass

if ttype == 'binary':
    scoring = 'roc_auc'
else:
    def scorer(est, X, y):
        return roc_auc_score(y, est.predict_proba(X), multi_class='ovr', average='macro')
    scoring = scorer

result = permutation_importance(
    pipeline, X_val, y_val, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)

imp = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std,
})
imp = imp.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
imp['importance_mean'] = imp['importance_mean'].round(4)
imp['importance_sd'] = imp['importance_sd'].round(4)

mlca_runtime.save_table(imp, 'interpretation.csv',
                        caption='Permutation importance on validation rows (roc_auc, 20 repeats)')
mlca_runtime.record_fact_candidate(
    'top_feature', str(imp.iloc[0]['feature']), source='interpretation.csv'
)
mlca_runtime.record_metric(
    'top_perm_importance_mean', float(imp.iloc[0]['importance_mean']), split='val'
)
print(f"top={imp.iloc[0]['feature']} mean={imp.iloc[0]['importance_mean']}")
