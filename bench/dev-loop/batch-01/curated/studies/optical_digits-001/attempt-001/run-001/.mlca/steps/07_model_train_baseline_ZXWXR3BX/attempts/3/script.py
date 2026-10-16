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
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

key_like = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if (s.dtype == 'object' or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)) and s.nunique() == len(s):
        key_like.append(col)

feature_cols = [c for c in train.columns if c not in drop and c not in key_like]
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
        return any(has_scaler(s) for s in obj.named_steps.values())
    if isinstance(obj, ColumnTransformer):
        return any(has_scaler(t) for _, t, _ in obj.transformers)
    return False

def contains(container, obj):
    if isinstance(container, Pipeline):
        return any(s is obj for s in container.named_steps.values())
    return False

if isinstance(selector, Pipeline) and contains(selector, preprocessor):
    base = selector
elif isinstance(preprocessor, Pipeline) and contains(preprocessor, selector):
    base = preprocessor
else:
    base = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

if not has_scaler(base):
    base = Pipeline(base.steps + [('scaler', StandardScaler())])

n_components = min(8, len(feature_cols))
full = Pipeline(base.steps + [('pca', PCA(n_components=n_components, random_state=42))])
fitted = mlca_runtime.fit_record(full, X_train, partition='train')
mlca_runtime.save_model(fitted, 'model.pkl')

pre_pca = Pipeline(fitted.steps[:-1])
Xv = pre_pca.transform(X_val)
pca = fitted.named_steps['pca']
Xr = pca.inverse_transform(pca.transform(Xv))
mse = float(((Xv - Xr) ** 2).mean().mean())
mlca_runtime.record_metric('reconstruction_error', mse, split='val')
mlca_runtime.record_metric('embedding_dimension', float(n_components), split='global')

print(f"Features: {len(feature_cols)}")
print(f"PCA components: {n_components}")
print(f"Val reconstruction error (scaled space): {mse:.4f}")
