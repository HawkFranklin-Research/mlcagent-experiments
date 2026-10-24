import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import is_classifier

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target_col = next(c for c in targets if c in train.columns)
y = train[target_col]

def load_pkl(key):
    p = inputs.get(key)
    if p is None:
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

def get_feature_names(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

latest = load_pkl('latest_model')
model_obj = load_pkl('08_model_train_logistic_487M7W52/model.pkl')
pre = load_pkl('06_preprocess_fit_preprocessing_Z6Z720YG/preprocessor.pkl')
sel = load_pkl('07_features_select_features_Y69JECM8/selector.pkl')

def pipeline_ok(obj):
    return (isinstance(obj, Pipeline) and obj.steps
            and is_classifier(obj.steps[-1][1])
            and get_feature_names(obj) is not None)

base = None
if pipeline_ok(latest):
    base = latest
elif pipeline_ok(model_obj):
    base = model_obj

new_est = HistGradientBoostingClassifier(random_state=seed)

if base is not None:
    feature_cols = get_feature_names(base)
    steps = list(base.steps)
    steps[-1] = (steps[-1][0], new_est)
    pipe = Pipeline(steps)
else:
    feature_cols = get_feature_names(pre) or get_feature_names(sel)
    steps = []
    if pre is not None:
        steps.append(('preprocessor', pre))
    if sel is not None:
        steps.append(('selector', sel))
    steps.append(('classifier', new_est))
    pipe = Pipeline(steps)

if feature_cols is None:
    exclude = set(targets) | set(excluded)
    if group_col:
        exclude.add(group_col)
    feature_cols = [c for c in train.columns if c not in exclude]
    keep = []
    for c in feature_cols:
        s = train[c]
        if s.dtype == 'object' and s.nunique() == len(s):
            continue
        keep.append(c)
    feature_cols = keep

bad = set(feature_cols) & (set(targets) | set(excluded))
if group_col:
    bad.add(group_col)
if bad:
    raise ValueError(f"Feature columns include target/excluded/group: {bad}")

X = train[feature_cols]
pipe = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print(f"Features: {len(feature_cols)}")
print(f"Train rows: {len(train)}")
print("Saved alternative_model.pkl")
