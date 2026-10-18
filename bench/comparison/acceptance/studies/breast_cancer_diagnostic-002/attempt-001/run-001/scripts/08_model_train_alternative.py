import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'task']:
    if k not in protocol:
        raise KeyError(f"protocol missing {k}; keys={list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if train[c].dtype == 'object' or str(train[c].dtype) == 'category':
        if train[c].nunique() == len(train) and train[c].notna().all():
            continue
    feature_cols.append(c)

with open(inputs['05_preprocess_fit_preprocessing_VDSDFD6K/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_C9CJJ4J0/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def raw_input_cols(obj):
    if isinstance(obj, Pipeline):
        return raw_input_cols(obj.steps[0][1])
    names = getattr(obj, 'feature_names_in_', None)
    return list(names) if names is not None else None

def accepts_raw(obj, raw_cols):
    cols = raw_input_cols(obj)
    return cols is not None and set(cols) == set(raw_cols)

sel_raw = accepts_raw(sel, feature_cols)
pre_raw = accepts_raw(pre, feature_cols)

if sel_raw:
    frontend = sel
    expected = raw_input_cols(sel)
elif pre_raw:
    frontend = Pipeline([('pre', pre), ('sel', sel)])
    expected = raw_input_cols(pre)
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])
    expected = raw_input_cols(pre)

if expected is not None:
    feature_cols = [c for c in expected if c in train.columns and c not in targets and c not in excluded and c != group_col]

try:
    frontend.set_output(transform='pandas')
except Exception:
    pass

counts = train[targets[0]].value_counts()
min_frac = counts.min() / counts.sum()
clf_kwargs = {'random_state': 42}
if min_frac < 0.10:
    clf_kwargs['class_weight'] = 'balanced'

clf = HistGradientBoostingClassifier(**clf_kwargs)
if isinstance(frontend, Pipeline):
    model = Pipeline(frontend.steps + [('clf', clf)])
else:
    model = Pipeline([('frontend', frontend), ('clf', clf)])

X_train = train[feature_cols]
y_train = train[targets[0]]
fit_kwargs = {}
if group_col and group_col in train.columns:
    fit_kwargs['groups'] = train[group_col].values

trained = mlca_runtime.fit_record(model, X_train, 'train', y=y_train, **fit_kwargs)
mlca_runtime.save_model(trained, 'alternative_model.pkl')

if primary_metric == 'roc_auc' and task['type'] == 'binary':
    X_val = val[feature_cols]
    y_val = val[targets[0]]
    proba = trained.predict_proba(X_val)
    classes = trained.classes_
    if len(classes) == 2:
        y_bin = (y_val == classes[1]).astype(int)
        auc = roc_auc_score(y_bin, proba[:, 1])
        mlca_runtime.record_metric('roc_auc', auc, 'val')
        print(f"val roc_auc={auc:.4f}")

print(f"n_features={len(feature_cols)}, train_shape={train.shape}")
