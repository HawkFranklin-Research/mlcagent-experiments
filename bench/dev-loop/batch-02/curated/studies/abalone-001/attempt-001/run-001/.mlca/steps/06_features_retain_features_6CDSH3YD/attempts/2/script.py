import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

with open(inputs['05_preprocess_fit_preprocessing_JT5JG4F1/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

feature_cols = None
if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
elif isinstance(preprocessor, Pipeline):
    for _, step in preprocessor.steps:
        if hasattr(step, 'feature_names_in_'):
            feature_cols = list(step.feature_names_in_)
            break

train = pd.read_parquet(inputs['train'])
if feature_cols is None:
    all_cols = train.columns.tolist()
    exclude = set(targets) | set(excluded)
    if group_col:
        exclude.add(group_col)
    feature_cols = [c for c in all_cols if c not in exclude]

X = train[feature_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

selector = FunctionTransformer(func=None)
if hasattr(selector, 'set_output'):
    selector.set_output(transform='pandas')

if isinstance(preprocessor, Pipeline):
    steps = preprocessor.steps + [('passthrough_selector', selector)]
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([('preprocessor', preprocessor), ('passthrough_selector', selector)])

if hasattr(pipeline, 'set_output'):
    pipeline.set_output(transform='pandas')

mlca_runtime.fit_record(pipeline, X, 'train', y)
mlca_runtime.save_model(pipeline, 'selector_all.pkl')

print(f"Fitted passthrough selector on {X.shape[0]} rows, {X.shape[1]} features")
