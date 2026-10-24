import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['latest_model'], 'rb') as f:
    latest_model = pickle.load(f)

if hasattr(latest_model, 'feature_names_in_'):
    feature_cols = list(latest_model.feature_names_in_)
elif isinstance(latest_model, Pipeline) and hasattr(latest_model.steps[0][1], 'feature_names_in_'):
    feature_cols = list(latest_model.steps[0][1].feature_names_in_)
else:
    raise ValueError("Cannot determine feature columns from model")

feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]

X_train = train[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

if isinstance(latest_model, Pipeline):
    steps = latest_model.steps
    last_name = steps[-1][0]
    new_steps = steps[:-1] + [(last_name, HistGradientBoostingClassifier(random_state=42))]
    new_pipeline = Pipeline(new_steps)
else:
    with open(inputs['05_preprocess_fit_preprocessing_R5N312M1/preprocessor.pkl'], 'rb') as f:
        preprocessor = pickle.load(f)
    with open(inputs['06_features_select_features_JQQFHDQ8/selector.pkl'], 'rb') as f:
        selector = pickle.load(f)
    new_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

fitted = mlca_runtime.fit_record(new_pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"Trained alternative model on {X_train.shape[0]} rows, {X_train.shape[1]} features")
