import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for key in required:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
group_cols = [group] if group else []

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_ZR2ARXZ1/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

def is_key_like(series):
    if series.dtype == 'object' or series.dtype.name == 'category' or series.dtype == 'string':
        return series.is_unique
    if pd.api.types.is_integer_dtype(series):
        return series.is_unique
    return False

all_cols = train.columns.tolist()
candidate_cols = [c for c in all_cols if c not in targets and c not in excluded and c not in group_cols]
feature_cols = [c for c in candidate_cols if not is_key_like(train[c])]

if hasattr(preprocessor, 'feature_names_in_'):
    expected = preprocessor.feature_names_in_.tolist()
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects columns not in train: {missing}")
    feature_cols = expected
elif isinstance(preprocessor, Pipeline):
    first_step = preprocessor.steps[0][1]
    if hasattr(first_step, 'feature_names_in_'):
        expected = first_step.feature_names_in_.tolist()
        missing = [c for c in expected if c not in train.columns]
        if missing:
            raise ValueError(f"Preprocessor expects columns not in train: {missing}")
        feature_cols = expected

for c in feature_cols:
    if c in targets or c in excluded or c in group_cols:
        raise ValueError(f"Feature column {c} is a target, excluded, or group column.")

X_train = train[feature_cols].copy()
y_train = train[targets[0]] if targets else None

selector = ColumnTransformer(
    transformers=[('passthrough', 'passthrough', slice(None))],
    remainder='drop'
)

if isinstance(preprocessor, Pipeline):
    steps = preprocessor.steps.copy()
    steps.append(('selector', selector))
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

pipeline.set_output(transform='pandas')

mlca_runtime.fit_record(pipeline, X_train, partition='train', y=y_train)

mlca_runtime.save_model(pipeline, 'selector_all.pkl')

print(f"Fitted selector on {X_train.shape[0]} rows, {X_train.shape[1]} features")
print("Saved selector_all.pkl")
