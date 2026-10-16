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
        raise KeyError(f"Missing '{k}'. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group_col = protocol.get('group_column', None)
task = protocol['task']
assert task.get('type') == 'multioutput_regression', task.get('type')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_ZW775027/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_1F75JGY7/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

exclude_set = set(targets) | set(excluded)
if group_col:
    exclude_set.add(group_col)
feature_cols = []
for c in train.columns:
    if c in exclude_set:
        continue
    s = train[c]
    if s.dtype.kind in 'iuO' and s.nunique(dropna=False) == len(train):
        continue
    feature_cols.append(c)

X_train = train[feature_cols].copy()
y_train = train[targets].copy()

def has_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return True
        if 'preprocess' in name.lower() or 'column' in name.lower():
            return True
    return False

if has_pre(selector):
    steps = list(selector.steps)
    if steps and hasattr(steps[-1][1], 'predict'):
        steps = steps[:-1]
    pipe = Pipeline(steps + [('reg', LinearRegression())])
else:
    steps = [('preprocessor', preprocessor)]
    if selector is not None:
        steps.append(('selector', selector))
    pipe = Pipeline(steps + [('reg', LinearRegression())])

try:
    pipe.set_output(transform='pandas')
except Exception:
    pass

model = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
mlca_runtime.save_model(model, 'model.pkl')
print(f"features={len(feature_cols)} targets={targets} rows={len(X_train)}")
