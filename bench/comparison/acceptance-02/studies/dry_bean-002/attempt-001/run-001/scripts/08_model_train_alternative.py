import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
with open(inputs['07_model_train_logistic_G7CT94CC/model.pkl'], 'rb') as f:
    old = pickle.load(f)

if hasattr(old, 'feature_names_in_'):
    feat_cols = [c for c in old.feature_names_in_ if c in train.columns]
else:
    feat_cols = [c for c in train.columns
                 if c not in targets and c not in excluded and c != group]

X = train[feat_cols]
y = train[targets[0]]

if isinstance(old, Pipeline):
    pre_steps = list(old.steps[:-1])
else:
    pre_steps = [('pre', old)]

model = Pipeline(pre_steps + [('clf', HistGradientBoostingClassifier(random_state=42))])
mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'alternative_model.pkl')
print('saved model; features', len(feat_cols), 'rows', len(train), 'steps', list(model.named_steps))
