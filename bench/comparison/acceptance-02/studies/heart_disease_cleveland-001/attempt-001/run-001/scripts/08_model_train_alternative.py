import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

seed = 42
mlca_runtime.set_seed(seed)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['07_model_train_logistic_12XC5SW7/model.pkl'], 'rb') as f:
    model = pickle.load(f)

if not isinstance(model, Pipeline):
    raise TypeError("Loaded model is not a sklearn Pipeline")

if hasattr(model, 'feature_names_in_'):
    feature_cols = model.feature_names_in_
else:
    first_step = model.steps[0][1]
    if hasattr(first_step, 'feature_names_in_'):
        feature_cols = first_step.feature_names_in_
    else:
        raise ValueError("Cannot determine feature columns from loaded model")

for col in targets + excluded:
    if col in feature_cols:
        raise ValueError(f"Target or excluded column {col} found in feature columns")

X_train = train[feature_cols]
y_train = train[targets[0]]

steps = list(model.steps)
steps[-1] = ('classifier', HistGradientBoostingClassifier(random_state=seed))
new_pipeline = Pipeline(steps)

mlca_runtime.fit_record(new_pipeline, X_train, 'train', y=y_train)

mlca_runtime.save_model(new_pipeline, 'alternative_model.pkl')

print(f"Trained HistGradientBoosting pipeline with {len(feature_cols)} features on {len(X_train)} rows.")
