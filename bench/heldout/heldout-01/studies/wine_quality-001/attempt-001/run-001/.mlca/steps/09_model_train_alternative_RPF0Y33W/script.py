import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
SEED = 42

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target = targets[0]

with open(inputs['08_model_train_logistic_5N3CJC0T/model.pkl'], 'rb') as f:
    base_model = pickle.load(f)

if isinstance(base_model, Pipeline) and len(base_model.steps) > 0:
    features = list(base_model.feature_names_in_) if hasattr(base_model, 'feature_names_in_') else [
        c for c in train.columns
        if c not in set(targets) | set(excluded) and c != group
    ]
    steps = [(name, clone(est)) for name, est in base_model.steps[:-1]]
    last_name = base_model.steps[-1][0]
    alt_model = Pipeline(steps + [(last_name, HistGradientBoostingClassifier(random_state=SEED))])
else:
    with open(inputs['06_preprocess_fit_preprocessing_1DA5F11K/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inputs['07_features_select_features_W8F8XFJ6/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    features = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else [
        c for c in train.columns
        if c not in set(targets) | set(excluded) and c != group
    ]
    has_selector = isinstance(pre, Pipeline) and any(
        n in ('select', 'selector', 'features') for n, _ in pre.steps
    )
    steps = [('preprocess', pre)]
    if not has_selector:
        steps.append(('select', sel))
    steps.append(('model', HistGradientBoostingClassifier(random_state=SEED)))
    alt_model = Pipeline(steps)

X_train = train[features]
y_train = train[target]

alt_model = mlca_runtime.fit_record(alt_model, X_train, 'train', y_train)
mlca_runtime.save_model(alt_model, 'alternative_model.pkl')
print('features', len(features), 'rows', len(X_train))
print('estimator', type(alt_model.steps[-1][1]).__name__ if isinstance(alt_model, Pipeline) else type(alt_model).__name__)
