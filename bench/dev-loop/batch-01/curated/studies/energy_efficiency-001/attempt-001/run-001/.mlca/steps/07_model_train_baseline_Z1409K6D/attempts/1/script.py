import mlca_runtime
import pickle
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing key '{k}' in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']
assert task.get('type') == 'multioutput_regression', f"Task type is {task.get('type')}"

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_ZW775027/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_1F75JGY7/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    exclude_set = set(targets) | set(excluded)
    if group_col:
        exclude_set.add(group_col)
    feature_cols = []
    for c in train.columns:
        if c in exclude_set:
            continue
        if train[c].nunique() == len(train) and train[c].dtype.kind in 'iuO':
            continue
        feature_cols.append(c)

feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and (group_col is None or c != group_col)]
feature_cols = [c for c in feature_cols if c in train.columns]

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
X_val = val[feature_cols].copy()
y_val = val[targets].copy()

def has_preprocessor(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return True
        if 'preprocess' in name.lower() or 'column' in name.lower():
            return True
    return False

if has_preprocessor(selector):
    base_steps = list(selector.steps)
    if base_steps and hasattr(base_steps[-1][1], 'predict'):
        base_steps = base_steps[:-1]
    final_pipeline = Pipeline(base_steps + [('reg', LinearRegression())])
else:
    base_steps = [('preprocessor', preprocessor)]
    if selector is not None:
        base_steps.append(('selector', selector))
    final_pipeline = Pipeline(base_steps + [('reg', LinearRegression())])

try:
    final_pipeline.set_output(transform='pandas')
except Exception:
    pass

model = mlca_runtime.fit_record(final_pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(model, 'model.pkl')
print(f"Fitted model with {len(feature_cols)} features; targets: {targets}")
