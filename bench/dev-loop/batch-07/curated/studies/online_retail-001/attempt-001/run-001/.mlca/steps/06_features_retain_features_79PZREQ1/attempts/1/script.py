import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_X6XZMNTX/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    all_cols = list(train.columns)
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = []
    for col in all_cols:
        if col in drop:
            continue
        if train[col].dtype in ['float64', 'float32']:
            feature_cols.append(col)
        elif train[col].dtype == 'object' or train[col].dtype.name.startswith('int'):
            if train[col].nunique() == len(train):
                continue
            feature_cols.append(col)
        else:
            feature_cols.append(col)

X_train = train[feature_cols].copy()

selector = FunctionTransformer(func=None, feature_names_out='one-to-one')

if isinstance(preprocessor, Pipeline):
    steps = preprocessor.steps + [('selector', selector)]
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

pipeline.set_output(transform='pandas')

fitted = mlca_runtime.fit_record(pipeline, X_train, partition='train')

mlca_runtime.save_model(fitted, 'selector_all.pkl')

print(f"X_train shape: {X_train.shape}")
print(f"Preprocessor type: {type(preprocessor).__name__}")
print(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
