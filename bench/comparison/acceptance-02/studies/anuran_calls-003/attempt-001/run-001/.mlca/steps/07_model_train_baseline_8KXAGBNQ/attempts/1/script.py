import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
missing = [k for k in ('targets', 'excluded_columns', 'task') if k not in protocol]
if missing:
    raise KeyError(f"missing protocol keys {missing}; available={list(protocol.keys())}")
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

pre = pickle.load(open(inputs['05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inputs['06_features_retain_features_2RG0T0WD/selector_all.pkl'], 'rb'))

def contains(outer, inner):
    if outer is inner:
        return True
    if isinstance(outer, Pipeline):
        return any(contains(s, inner) for s in outer.named_steps.values())
    return False

if isinstance(sel, Pipeline) and contains(sel, pre):
    base = list(sel.steps)
elif isinstance(pre, Pipeline) and contains(pre, sel):
    base = list(pre.steps)
else:
    base = [('pre', pre), ('sel', sel)]

model = Pipeline(base + [('clf', MultiOutputClassifier(LogisticRegression(max_iter=2000)))])

first = model.steps[0][1]
if hasattr(first, 'feature_names_in_'):
    feat_cols = list(first.feature_names_in_)
else:
    feat_cols = list(train.columns)
feat_cols = [c for c in feat_cols if c in train.columns and c not in targets and c not in excluded and c != group]

X_train = train[feat_cols]
X_val = val[feat_cols]
y_train = train[targets]
y_val = val[targets]
print('n_features', len(feat_cols), 'n_train', len(X_train))

model = mlca_runtime.fit_record(model, X_train, 'train', y=y_train)

yhat = model.predict(X_val)
f1s = [float(f1_score(y_val.iloc[:, i], yhat[:, i], average='macro')) for i in range(len(targets))]
macro = float(np.mean(f1s))
mlca_runtime.record_metric('macro_f1', macro, 'val')
print('val macro_f1', round(macro, 4))

mlca_runtime.save_model(model, 'model.pkl')
