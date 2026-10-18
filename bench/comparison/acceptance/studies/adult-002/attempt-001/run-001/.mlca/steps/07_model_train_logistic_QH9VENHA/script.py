import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_HP31G2ZF/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

def get_step_names(obj):
    if isinstance(obj, Pipeline):
        return list(obj.named_steps.keys())
    return []

def get_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = list(obj.named_steps.values())[0]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

pre_cols = get_input_cols(preprocessor) or get_input_cols(selector)
if pre_cols is None:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    pre_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object and s.nunique(dropna=True) == len(s):
            continue
        pre_cols.append(c)

selector_steps = [s.lower() for s in get_step_names(selector)]
selector_has_preproc = any('preproc' in s for s in selector_steps)

if selector_has_preproc:
    combined = selector
else:
    combined = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

final = Pipeline([
    ('combined', combined),
    ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
])

X_train = train[pre_cols]
y_train = train[targets[0]]

mlca_runtime.fit_record(final, X_train, 'train', y_train)
mlca_runtime.save_model(final, 'model.pkl')
print(f"features: {len(pre_cols)}; metric: {primary_metric}; rows: {len(X_train)}; selector_has_preproc: {selector_has_preproc}")
