import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in candidate_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train) and train[c].notna().all():
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train) and train[c].notna().all():
            key_like.append(c)
feature_cols = [c for c in candidate_cols if c not in key_like]

with open(inputs['05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_P9FZ0GCB/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

selector_includes_preprocessor = False
if isinstance(selector, Pipeline):
    if 'preprocessor' in selector.named_steps:
        selector_includes_preprocessor = True
    elif any(step is preprocessor for step in selector.named_steps.values()):
        selector_includes_preprocessor = True
    else:
        first_step = list(selector.named_steps.values())[0]
        if first_step.__class__.__name__ == 'ColumnTransformer':
            selector_includes_preprocessor = True

if hasattr(preprocessor, 'feature_names_in_'):
    expected = list(preprocessor.feature_names_in_)
    if set(expected) != set(feature_cols):
        raise ValueError("Feature columns mismatch with preprocessor")
    feature_cols = expected

if selector_includes_preprocessor and hasattr(selector, 'feature_names_in_'):
    expected = list(selector.feature_names_in_)
    if set(expected) != set(feature_cols):
        raise ValueError("Feature columns mismatch with selector")
    feature_cols = expected

n_rows = len(train)
solver = 'saga' if n_rows > 100000 else 'lbfgs'
model = MultiOutputClassifier(LogisticRegression(max_iter=2000, solver=solver))

if selector_includes_preprocessor:
    pipeline = Pipeline([
        ('selector', selector),
        ('model', model)
    ])
else:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', model)
    ])

X = train[feature_cols]
y = train[targets]

fitted = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Targets: {targets}")
print(f"Pipeline steps: {list(fitted.named_steps.keys())}")
