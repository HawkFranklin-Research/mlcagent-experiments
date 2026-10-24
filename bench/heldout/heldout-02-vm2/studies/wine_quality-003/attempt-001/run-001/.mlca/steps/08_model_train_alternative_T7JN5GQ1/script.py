import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42); seed = 42
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
for c in list(feature_cols):
    s = train[c]
    if s.dtype == object and s.nunique() == len(train) and s.astype(str).str.len().max() < 32:
        feature_cols.remove(c)

with open(inp['05_preprocess_fit_preprocessing_HXN3H5DR/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_ANJ3XYKV/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)
with open(inp['07_model_train_logistic_5W0X2YFF/model.pkl'], 'rb') as f:
    mdl = pickle.load(f)

y = train[targets[0]]
kw = {'class_weight': 'balanced'} if y.value_counts(normalize=True).min() < 0.10 else {}
hgb = HistGradientBoostingClassifier(random_state=seed, **kw)

if isinstance(mdl, Pipeline) and hasattr(mdl, 'feature_names_in_'):
    pipe = Pipeline(list(mdl.steps[:-1]) + [('model', hgb)])
    raw_cols = list(mdl.feature_names_in_)
elif hasattr(sel, 'feature_names_in_') and set(sel.feature_names_in_) == set(feature_cols):
    pipe = Pipeline([('selector', sel), ('model', hgb)])
    raw_cols = list(sel.feature_names_in_)
else:
    comps = [('preprocessor', pre)]
    if hasattr(sel, 'transform'):
        comps.append(('selector', sel))
    comps.append(('model', hgb))
    pipe = Pipeline(comps)
    raw_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else feature_cols

Xtr = train[raw_cols]
pipe = mlca_runtime.fit_record(pipe, Xtr, 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

Xv = val[raw_cols]; yv = val[targets[0]]
proba = pipe.predict_proba(Xv)
classes = list(pipe.classes_)
aucs = []
for c in classes:
    if (yv == c).any():
        j = classes.index(c)
        aucs.append(roc_auc_score((yv == c).astype(int), proba[:, j]))
macro = float(np.mean(aucs))
mlca_runtime.record_metric('val_roc_auc_ovr_macro', macro, 'val')
print('train', train.shape, 'val', val.shape, 'features', len(raw_cols), 'macro_auc', round(macro, 4))
print('classes', classes)
