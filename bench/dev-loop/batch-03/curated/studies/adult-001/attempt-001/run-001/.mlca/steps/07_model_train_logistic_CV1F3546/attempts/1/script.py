import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group_col = protocol.get('group_column')
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available keys: {list(protocol.keys())}")

train = pd.read_parquet(inputs['train'])

all_cols = train.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

def is_key_like(series):
    n = len(series)
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return series.nunique() == n
    if pd.api.types.is_integer_dtype(series):
        return series.nunique() == n and (series.max() - series.min() + 1) == n
    return False

key_like = [c for c in feature_cols if is_key_like(train[c])]
feature_cols = [c for c in feature_cols if c not in key_like]

with open(inputs['05_preprocess_fit_preprocessing_F5B2KN9D/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_Y4ZJEG70/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

def is_raw_frontend(obj, cols):
    if hasattr(obj, 'feature_names_in_'):
        return set(obj.feature_names_in_) == set(cols)
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return True
            if hasattr(step, 'feature_names_in_') and set(step.feature_names_in_) == set(cols):
                return True
    return False

if is_raw_frontend(selector, feature_cols):
    if hasattr(selector, 'feature_names_in_'):
        feature_cols = list(selector.feature_names_in_)
    pipeline = Pipeline([
        ('selector', selector),
        ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42))
    ])
else:
    if hasattr(preprocessor, 'feature_names_in_'):
        feature_cols = list(preprocessor.feature_names_in_)
    if hasattr(preprocessor, 'set_output'):
        try:
            sample = train[feature_cols].head(5)
            out = preprocessor.transform(sample)
            n_cols = out.shape[1] if hasattr(out, 'shape') else len(out[0])
            if n_cols <= 300:
                preprocessor.set_output(transform='pandas')
        except Exception:
            pass
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42))
    ])

target = targets[0]
fitted = mlca_runtime.fit_record(pipeline, X=train[feature_cols], partition='train', y=train[target])
mlca_runtime.save_model(fitted, 'model.pkl')
print(f"Train shape: {train.shape}")
print(f"Feature count: {len(feature_cols)}")
print("Model saved: model.pkl")
