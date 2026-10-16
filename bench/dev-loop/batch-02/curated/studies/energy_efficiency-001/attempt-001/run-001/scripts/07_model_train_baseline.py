import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
def get_protocol(key, default=None):
    if key not in protocol:
        if default is not None:
            return default
        raise KeyError(f"Missing '{key}' in protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol('targets')
excluded = get_protocol('excluded_columns', [])
group_col = get_protocol('group_column', None)
task = get_protocol('task')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_8F2KVPF9/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def get_feature_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

def has_preprocessing(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return True
            if 'preprocess' in name.lower():
                return True
            if has_preprocessing(step):
                return True
    return False

selector_has_preproc = has_preprocessing(selector) if selector is not None else False

if selector_has_preproc:
    raw_features = get_feature_names(selector)
else:
    raw_features = get_feature_names(preprocessor)

if raw_features is None:
    exclude_set = set(targets) | set(excluded)
    if group_col:
        exclude_set.add(group_col)
    candidate = [c for c in train.columns if c not in exclude_set]
    raw_features = []
    for c in candidate:
        col = train[c]
        if col.dtype == object or pd.api.types.is_string_dtype(col.dtype):
            if col.nunique() == len(train):
                continue
        elif pd.api.types.is_integer_dtype(col.dtype):
            if col.nunique() == len(train):
                vals = col.dropna().unique()
                if len(vals) == len(train) and col.max() - col.min() + 1 == len(train):
                    continue
        raw_features.append(c)

for c in raw_features:
    if c in targets or c in excluded or (group_col and c == group_col):
        raise ValueError(f"Feature {c} is a target/excluded/group column")

model = LinearRegression()
if selector_has_preproc:
    if isinstance(selector, Pipeline):
        steps = list(selector.steps) + [('model', model)]
        final_pipeline = Pipeline(steps)
    else:
        final_pipeline = Pipeline([('selector', selector), ('model', model)])
else:
    steps = [('preprocessor', preprocessor)]
    if selector is not None:
        steps.append(('selector', selector))
    steps.append(('model', model))
    final_pipeline = Pipeline(steps)

X_train = train[raw_features].copy()
y_train = train[targets].copy()
X_val = val[raw_features].copy()
y_val = val[targets].copy()

fitted = mlca_runtime.fit_record(final_pipeline, X_train, 'train', y_train)

pred = np.asarray(fitted.predict(X_val))
rmse_list = []
for i, t in enumerate(targets):
    rmse = np.sqrt(np.mean((pred[:, i] - y_val[t].values) ** 2))
    mlca_runtime.record_metric(f'rmse_{t}', float(rmse), 'val')
    rmse_list.append(rmse)
mean_rmse = float(np.mean(rmse_list))
mlca_runtime.record_metric('rmse_mean', mean_rmse, 'val')

mlca_runtime.save_model(fitted, "model.pkl")

print(f"Targets: {targets}")
print(f"Train: {train.shape}, Val: {val.shape}")
print(f"Val RMSE mean: {mean_rmse:.4f}")
