import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
target = targets[0]
task = protocol['task']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['07_model_train_logistic_RQ5H03SW/model.pkl'], 'rb') as f:
    base = pickle.load(f)

if not isinstance(base, Pipeline):
    raise RuntimeError('Expected a Pipeline in saved model.pkl')

first_step = base.steps[0][1]
if hasattr(first_step, 'feature_names_in_'):
    feature_cols = list(first_step.feature_names_in_)
else:
    excluded = set(protocol.get('excluded_columns', [])) | set(targets)
    if group_col:
        excluded.add(group_col)
    feature_cols = [c for c in train.columns if c not in excluded]

X = train[feature_cols]
y = train[target]

steps = list(base.steps)
clf_kwargs = {'random_state': seed}
if y.value_counts(normalize=True).min() < 0.10:
    clf_kwargs['class_weight'] = 'balanced'
steps[-1] = ('classifier', HistGradientBoostingClassifier(**clf_kwargs))
alt_pipeline = Pipeline(steps)

fitted = mlca_runtime.fit_record(alt_pipeline, X, partition='train', y=y)
mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f'Features: {len(feature_cols)}, Train shape: {X.shape}')
print(f'Class balance: {y.value_counts(normalize=True).to_dict()}')
