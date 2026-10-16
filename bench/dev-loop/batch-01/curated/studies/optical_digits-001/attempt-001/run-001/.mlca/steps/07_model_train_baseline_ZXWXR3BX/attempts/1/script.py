import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

all_cols = list(train.columns)
drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

key_like = []
for col in all_cols:
    if col in drop_cols:
        continue
    s = train[col]
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s.dtype):
        if s.nunique() == len(s):
            key_like.append(col)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique() == len(s):
            key_like.append(col)

feature_cols = [c for c in all_cols if c not in drop_cols and c not in key_like]
feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

with open(inputs['05_preprocess_fit_preprocessing_S0HRN8VM/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_5F8P6A23/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def has_scaler(obj):
    if isinstance(obj, (StandardScaler, MinMaxScaler, RobustScaler)):
        return True
    if isinstance(obj, Pipeline):
        return any(has_scaler(step) for step in obj.named_steps.values())
    if isinstance(obj, ColumnTransformer):
        return any(has_scaler(trans) for _, trans, _ in obj.transformers)
    return False

def contains(container, obj):
    if isinstance(container, Pipeline):
        return any(step is obj for step in container.named_steps.values())
    return False

if isinstance(selector, Pipeline) and contains(selector, preprocessor):
    base = selector
elif isinstance(preprocessor, Pipeline) and contains(preprocessor, selector):
    base = preprocessor
elif isinstance(selector, Pipeline):
    base = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
elif isinstance(preprocessor, Pipeline):
    base = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
else:
    base = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

if not has_scaler(base):
    base = Pipeline(base.steps + [('scaler', StandardScaler())])

n_features = len(feature_cols)
n_components = min(8, n_features)
pca = PCA(n_components=n_components, random_state=42)
full_pipeline = Pipeline(base.steps + [('pca', pca)])

fitted = mlca_runtime.fit_record(full_pipeline, X_train, partition='train')
mlca_runtime.save_model(fitted, 'model.pkl')

steps_before_pca = fitted.steps[:-1]
pipeline_before_pca = Pipeline(steps_before_pca)
X_val_scaled = pipeline_before_pca.transform(X_val)
pca_fitted = fitted.named_steps['pca']
X_val_pca = pca_fitted.transform(X_val_scaled)
X_val_recon = pca_fitted.inverse_transform(X_val_pca)
mse = ((X_val_scaled - X_val_recon) ** 2).mean().mean()
mlca_runtime.record_metric('reconstruction_error', float(mse), split='val')
mlca_runtime.record_metric('embedding_dimension', n_components, split='global')

print(f"Features: {len(feature_cols)}")
print(f"PCA components: {n_components}")
print(f"Val reconstruction error: {mse:.4f}")
