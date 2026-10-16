import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_Z1YWJ829/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_M3TS3EMK/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def get_feature_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

if isinstance(preprocessor, Pipeline) and 'selector' in preprocessor.named_steps:
    frontend = preprocessor
elif isinstance(selector, Pipeline) and 'preprocessor' in selector.named_steps:
    frontend = selector
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

feature_cols = get_feature_names(frontend)
if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        if train[c].dtype == 'object' or str(train[c].dtype) == 'category':
            if train[c].nunique() == len(train):
                feature_cols.remove(c)
        elif pd.api.types.is_integer_dtype(train[c]):
            if train[c].nunique() == len(train) and train[c].min() >= 0:
                feature_cols.remove(c)

X = train[feature_cols]
y = train[targets].copy()
for col in y.columns:
    if y[col].dtype == 'object' or str(y[col].dtype) == 'category':
        y[col] = y[col].astype('category').cat.codes

pipeline = Pipeline([
    ('frontend', frontend),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, 'alternative_model.pkl')
