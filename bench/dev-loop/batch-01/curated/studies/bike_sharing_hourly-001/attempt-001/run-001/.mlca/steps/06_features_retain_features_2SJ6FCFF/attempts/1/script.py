import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])

feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or col == group:
        continue
    if train[col].dtype == 'object' or train[col].dtype == 'string':
        if train[col].nunique() == len(train):
            continue
    elif pd.api.types.is_integer_dtype(train[col]):
        if train[col].nunique() == len(train):
            continue
    feature_cols.append(col)

with open(inputs['05_preprocess_fit_preprocessing_K60ASHDR/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if isinstance(preprocessor, Pipeline):
    steps = preprocessor.steps
else:
    steps = [('preprocessor', preprocessor)]

selector = ColumnTransformer(transformers=[], remainder='passthrough')
pipeline = Pipeline(steps + [('selector', selector)])

X = train[feature_cols]
y = train[targets[0]] if targets else None

mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, 'selector_all.pkl')

print(f"Features: {len(feature_cols)}")
print(f"Train shape: {X.shape}")
print("Saved selector_all.pkl")
