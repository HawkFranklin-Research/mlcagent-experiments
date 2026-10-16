import mlca_runtime
import pandas as pd
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

mlca_runtime.set_seed(42)

class PassthroughSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else None
        return self
    def transform(self, X):
        return X
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if self.feature_names_in_ is not None else input_features

def get_protocol_key(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = get_protocol_key(protocol, 'targets')
excluded = get_protocol_key(protocol, 'excluded_columns')
group = get_protocol_key(protocol, 'group_column')
task = get_protocol_key(protocol, 'task')

train = pd.read_parquet(inputs['train'])

drop_cols = set(targets + excluded)
if group and group in train.columns:
    drop_cols.add(group)

candidate_cols = [c for c in train.columns if c not in drop_cols]

def is_key_like(series):
    if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
        return series.nunique() == len(series)
    if pd.api.types.is_integer_dtype(series):
        return series.nunique() == len(series)
    return False

feature_cols = [c for c in candidate_cols if not is_key_like(train[c])]

X_train = train[feature_cols].copy()

with open(inputs['05_preprocess_fit_preprocessing_ZW775027/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'set_output'):
    preprocessor.set_output(transform='pandas')

X_train_pre = preprocessor.transform(X_train)

if not isinstance(X_train_pre, pd.DataFrame):
    if hasattr(preprocessor, 'get_feature_names_out'):
        X_train_pre = pd.DataFrame(
            X_train_pre,
            columns=preprocessor.get_feature_names_out(),
            index=X_train.index
        )

selector = PassthroughSelector()
mlca_runtime.fit_record(selector, X_train_pre, 'train')

step_name = 'selector'
if isinstance(preprocessor, Pipeline) and step_name in preprocessor.named_steps:
    step_name = 'passthrough_selector'

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps) + [(step_name, selector)]
    full_pipeline = Pipeline(steps)
else:
    full_pipeline = Pipeline([('preprocessor', preprocessor), (step_name, selector)])

mlca_runtime.save_model(full_pipeline, 'selector_all.pkl')

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Preprocessed shape: {X_train_pre.shape}")
print("Saved selector_all.pkl")
