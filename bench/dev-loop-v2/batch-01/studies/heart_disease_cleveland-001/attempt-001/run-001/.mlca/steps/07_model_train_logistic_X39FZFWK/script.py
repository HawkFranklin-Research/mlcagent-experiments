import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_7578M7XP/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_J83AE8FG/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def input_cols(obj):
    if obj is None:
        return None
    names = getattr(obj, 'feature_names_in_', None)
    if names is not None:
        return list(names)
    if hasattr(obj, 'named_steps') and obj.named_steps:
        return input_cols(list(obj.named_steps.values())[0])
    return None

feature_cols = input_cols(pre) or input_cols(sel)
if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        s = train[c]
        if s.nunique(dropna=False) == len(train) and (
            s.dtype == object or pd.api.types.is_integer_dtype(s)
        ):
            feature_cols.remove(c)

assert set(feature_cols) <= set(train.columns)
assert target not in feature_cols
assert not (set(excluded) & set(feature_cols))
if group_col:
    assert group_col not in feature_cols

X = train[feature_cols].copy()
y = train[target]

sel_cols = input_cols(sel)
if sel is not None and sel_cols == feature_cols:
    frontend = sel if hasattr(sel, 'named_steps') else (
        Pipeline([('pre', pre), ('sel', sel)]) if pre is not None else sel
    )
elif sel is not None and pre is not None:
    frontend = Pipeline([('pre', pre), ('sel', sel)])
elif pre is not None:
    frontend = pre
else:
    frontend = 'passthrough'

steps = []
if frontend != 'passthrough':
    steps.append(('features', frontend))
steps.append(('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)))
model = Pipeline(steps)

fitted = mlca_runtime.fit_record(model, X, 'train', y=y)

if hasattr(fitted, 'predict_proba') and len(np.unique(y)) == 2:
    proba = fitted.predict_proba(X)[:, 1]
    pos = fitted.classes_[1]
    y_bin = (y == pos).astype(int)
    auc = roc_auc_score(y_bin, proba)
    mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')

mlca_runtime.save_model(fitted, 'model.pkl')
print(f'train rows: {len(X)}, features: {len(feature_cols)}')
