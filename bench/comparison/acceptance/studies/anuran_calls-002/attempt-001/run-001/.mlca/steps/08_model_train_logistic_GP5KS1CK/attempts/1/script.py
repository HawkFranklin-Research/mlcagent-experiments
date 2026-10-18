import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol['task']
primary = task['primary_metric']

train = pd.read_parquet(inp['train'])

with open(inp['06_preprocess_fit_preprocessing_MYPQXFW6/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['07_features_select_features_NK102NGP/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def get_names(obj):
    return getattr(obj, 'feature_names_in_', None)

def get_width(obj):
    try:
        return len(obj.get_feature_names_out())
    except Exception:
        return None

sel_wraps = False
if isinstance(sel, Pipeline):
    for _, step in sel.named_steps.items():
        if step is pre or step.__class__.__name__ == 'ColumnTransformer':
            sel_wraps = True
            break
        if isinstance(step, Pipeline):
            for _, s2 in step.named_steps.items():
                if s2 is pre or s2.__class__.__name__ == 'ColumnTransformer':
                    sel_wraps = True
                    break
pre_in = get_names(pre)
sel_in = get_names(sel)
if sel_in is not None and pre_in is not None:
    if np.array_equal(np.asarray(sel_in), np.asarray(pre_in)):
        sel_wraps = True

lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_wraps:
    pipeline = Pipeline([('frontend', sel), ('model', lr)])
    expected = sel_in if sel_in is not None else pre_in
    width = get_width(sel)
else:
    steps = [('preprocessor', pre)]
    if isinstance(sel, Pipeline):
        steps += list(sel.steps)
    else:
        steps.append(('selector', sel))
    steps.append(('model', lr))
    pipeline = Pipeline(steps)
    expected = pre_in
    width = get_width(sel)
    if width is None:
        width = get_width(pre)

if width is not None and width <= 300:
    try:
        pipeline.set_output(transform='pandas')
    except Exception:
        pass

exclude = set(targets) | set(excluded) | ({group} if group else set())
if expected is not None:
    feature_cols = [c for c in expected if c not in exclude]
else:
    feature_cols = [c for c in train.columns if c not in exclude]
    for c in list(feature_cols):
        s = train[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            feature_cols.remove(c)

X = train[feature_cols].copy()
y = train[targets[0]].copy()

fitted = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')

if len(X) > 100000:
    idx = X.sample(n=10000, random_state=42).index
    Xe, ye = X.loc[idx], y.loc[idx]
    print('subsample', len(idx))
else:
    Xe, ye = X, y

proba = fitted.predict_proba(Xe)
auc = roc_auc_score(ye, proba, multi_class='ovr', average='macro', labels=fitted.classes_)
mlca_runtime.record_metric('train_roc_auc_ovr', float(auc), 'train')
print('rows', len(X), 'features', X.shape[1])
print('classes', len(fitted.classes_))
print('train_roc_auc_ovr', round(float(auc), 4))
