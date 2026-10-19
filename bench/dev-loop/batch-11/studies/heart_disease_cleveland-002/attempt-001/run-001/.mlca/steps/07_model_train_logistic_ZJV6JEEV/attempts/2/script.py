import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')
task = protocol['task']
assert task.get('type') == 'binary'

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_FQWKV8QW/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_6RNK7ZHK/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)


def input_cols(obj):
    fn = getattr(obj, 'feature_names_in_', None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline) and obj.named_steps:
        return input_cols(list(obj.named_steps.values())[0])
    return None


def has_step_of_type(obj, other):
    if not isinstance(obj, Pipeline):
        return False
    for step in obj.named_steps.values():
        if type(step) is type(other):
            return True
        if isinstance(step, Pipeline) and has_step_of_type(step, other):
            return True
    return False


pre_in = input_cols(preprocessor)
sel_in = input_cols(selector)

if pre_in is not None and sel_in is not None and list(sel_in) == list(pre_in):
    mode = 'selector_only'
    feat = sel_in
elif pre_in is not None and sel_in is not None:
    mode = 'chain'
    feat = pre_in
elif has_step_of_type(selector, preprocessor):
    mode = 'selector_only'
    feat = pre_in or sel_in
else:
    mode = 'chain'
    feat = pre_in or sel_in

if feat is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feat = [c for c in train.columns if c not in drop]

feat = [c for c in feat if c in train.columns]
X = train[list(feat)].copy()
y = train[targets[0]]

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if mode == 'selector_only':
    pipeline = Pipeline([('selector', selector), ('clf', clf)])
else:
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('selector', selector),
                         ('clf', clf)])

mlca_runtime.fit_record(pipeline, X, 'train', y)
mlca_runtime.save_model(pipeline, 'model.pkl')
mlca_runtime.record_metric('n_train_rows', float(len(X)), 'train')
print('rows', len(X), 'features', len(feat), 'mode', mode)
