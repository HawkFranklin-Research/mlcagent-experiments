import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')

def key_like(s):
    n, nn = s.nunique(dropna=True), int(s.notna().sum())
    if n != nn or n <= 1:
        return False
    return s.dtype == object or pd.api.types.is_integer_dtype(s.dtype)

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group
                and not key_like(train[c])]
raw = set(feature_cols)

pre = pickle.loads(inputs['05_preprocess_fit_preprocessing_QB1BRK7H/preprocessor.pkl'].read_bytes())
sel = pickle.loads(inputs['06_features_select_features_6NMQRFNK/selector.pkl'].read_bytes())
json.loads(inputs['05_preprocess_fit_preprocessing_QB1BRK7H/manifest.json'].read_text())
json.loads(inputs['06_features_select_features_6NMQRFNK/manifest.json'].read_text())

def input_names(obj):
    steps = list(obj.named_steps.values()) if isinstance(obj, Pipeline) else [obj]
    for o in steps:
        n = getattr(o, 'feature_names_in_', None)
        if n is not None:
            return list(n)
    return None

pre_in, sel_in = input_names(pre), input_names(sel)

if sel_in is not None and set(sel_in) <= raw:
    feature_cols = list(sel_in)
    frontend = sel
else:
    if pre_in is not None and set(pre_in) <= raw:
        feature_cols = list(pre_in)
    frontend = Pipeline([('preprocessor', pre), ('selector', sel)])

model = Pipeline([('frontend', frontend),
                  ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))])
X, y = train[feature_cols], train[targets[0]]
fitted = mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')
mlca_runtime.record_metric('n_features', float(len(feature_cols)), 'train')
mlca_runtime.record_metric('n_train_rows', float(len(train)), 'train')
print('features', len(feature_cols))
print('train', train.shape, 'val', val.shape)
print('frontend', type(frontend).__name__)
print('saved model.pkl')
