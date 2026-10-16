import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])

with open(inputs['07_model_train_baseline_5MMHKFNM/model.pkl'], 'rb') as f:
    baseline = pickle.load(f)

if isinstance(baseline, Pipeline):
    pipeline = clone(baseline)
    pipeline.steps[-1] = ('model', RandomForestRegressor(n_estimators=100, random_state=42))
else:
    with open(inputs['05_preprocess_fit_preprocessing_DVNTE848/preprocessor.pkl'], 'rb') as f:
        preprocessor = pickle.load(f)
    with open(inputs['06_features_retain_features_TBVVBD2H/selector_all.pkl'], 'rb') as f:
        selector = pickle.load(f)
    if isinstance(selector, Pipeline) and any(name == 'preprocessor' for name, _ in selector.steps):
        pipeline = Pipeline([('selector', selector), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    else:
        pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

feature_cols = None
if hasattr(pipeline, 'feature_names_in_'):
    feature_cols = list(pipeline.feature_names_in_)
else:
    for name, step in pipeline.steps:
        if hasattr(step, 'feature_names_in_'):
            feature_cols = list(step.feature_names_in_)
            break
if feature_cols is None:
    drop_cols = set(targets) | set(excluded)
    if group_col:
        drop_cols.add(group_col)
    feature_cols = [c for c in train.columns if c not in drop_cols]

feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]
X_train = train[feature_cols].copy()
y_train = train[targets].copy()

mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipeline, 'alternative_model.pkl')
print("Fitted alternative model")
print("Saved alternative_model.pkl")
