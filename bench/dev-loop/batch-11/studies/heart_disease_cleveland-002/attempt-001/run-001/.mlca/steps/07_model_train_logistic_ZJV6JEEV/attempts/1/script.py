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

def raw_input_cols(obj):
    fn = getattr(obj, 'feature_names_in_', None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline) and obj.named_steps:
        first = list(obj.named_steps.values())[0]
        return raw_input_cols(first)
    return None

feat = raw_input_cols(preprocessor)
if feat is None:
    feat = raw_input_cols(selector)
if feat is not None:
    feat = [c for c in feat if c in train.columns]

if feat is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    cand = [c for c in train.columns if c not in drop]
    n = len(train)
    feat = []
    for c in cand:
        s = train[c]
        if s.dtype == object:
            u = s.dropna().unique()
            if len(u) == n and int(s.isna().sum()) == 0:
                continue
        if pd.api.types.is_integer_dtype(s) and int(s.isna().sum()) == 0 \
                and s.nunique() == n and n > 1:
            continue
        feat.append(c)

X = train[list(feat)].copy()
y = train[targets[0]]

def contains(pipe, target):
    try:
        if isinstance(pipe, Pipeline):
            return any(step is target for step in pipe.named_steps.values())
    except Exception:
        pass
    return False

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if contains(selector, preprocessor):
    pipeline = Pipeline([('selector', selector), ('clf', clf)])
elif contains(preprocessor, selector):
    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
else:
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('selector', selector),
                         ('clf', clf)])

mlca_runtime.fit_record(pipeline, X, 'train', y)
mlca_runtime.save_model(pipeline, 'model.pkl')
mlca_runtime.record_metric('n_train_rows', float(len(X)), 'train')
print('rows', len(X), 'features', len(feat))
