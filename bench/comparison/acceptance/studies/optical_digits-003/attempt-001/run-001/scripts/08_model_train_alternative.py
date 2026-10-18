import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing '{key}' in protocol. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)
primary_metric = protocol['task']['primary_metric']

feature_columns = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
for c in list(feature_columns):
    if train[c].dtype == 'object' and train[c].nunique() == len(train):
        feature_columns.remove(c)

with open(inputs['07_model_train_baseline_V6BDRB4H/model.pkl'], 'rb') as f:
    baseline = pickle.load(f)

def get_frontend_steps(obj):
    if isinstance(obj, Pipeline):
        steps = list(obj.steps)
        if steps and hasattr(steps[-1][1], 'predict'):
            steps = steps[:-1]
        return steps
    return [('frontend', obj)]

frontend_steps = get_frontend_steps(baseline)
frontend_pipeline = Pipeline(steps=frontend_steps)
if hasattr(frontend_pipeline, 'feature_names_in_'):
    expected = list(frontend_pipeline.feature_names_in_)
    for c in expected:
        if c in targets or c in excluded or c == group_col:
            raise ValueError(f"Frontend expects excluded column {c}")
    feature_columns = expected

X_train = train[feature_columns]
X_val = val[feature_columns]
X_train_processed = frontend_pipeline.transform(X_train)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
full_pipeline = Pipeline(steps=frontend_steps + [('mlp', mlp)])
mlca_runtime.fit_record(full_pipeline, X_train, partition='train', y=X_train_processed)
mlca_runtime.save_model(full_pipeline, 'alternative_model.pkl')

X_val_processed = full_pipeline[:-1].transform(X_val)
pred_val = full_pipeline.predict(X_val)
mse_val = mean_squared_error(X_val_processed, pred_val)
mlca_runtime.record_metric(primary_metric, mse_val, split='val')

print(f"Features: {len(feature_columns)}")
print(f"Train shape: {X_train.shape}")
print(f"Val reconstruction MSE: {mse_val:.4f}")
