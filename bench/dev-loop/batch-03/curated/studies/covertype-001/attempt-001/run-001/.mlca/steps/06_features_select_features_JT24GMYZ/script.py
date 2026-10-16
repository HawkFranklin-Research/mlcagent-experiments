import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val']) if 'val' in inputs else None

with open(inputs['_protocol'], 'r') as f:
    protocol = json.load(f)

def get_protocol(key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol('targets')
excluded = get_protocol('excluded_columns')
group = get_protocol('group_column')
task = get_protocol('task')

with open(inputs['05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

with open(inputs['05_preprocess_fit_preprocessing_1A6AJYGR/manifest.json'], 'r') as f:
    manifest = json.load(f)

feature_cols = None
if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
elif 'feature_columns' in manifest:
    feature_cols = manifest['feature_columns']
elif 'input_columns' in manifest:
    feature_cols = manifest['input_columns']
else:
    all_cols = list(train.columns)
    drop = set(targets) | set(excluded)
    if group and group in all_cols:
        drop.add(group)
    feature_cols = [c for c in all_cols if c not in drop]
    key_like = [c for c in feature_cols if train[c].dtype == 'object' and train[c].nunique() == len(train)]
    feature_cols = [c for c in feature_cols if c not in key_like]

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Feature columns missing from train: {missing}")

X_train = train[feature_cols].copy()
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

n_out = None
if hasattr(preprocessor, 'get_feature_names_out'):
    try:
        n_out = len(preprocessor.get_feature_names_out())
    except Exception:
        pass
if n_out is None:
    if 'output_columns' in manifest:
        n_out = len(manifest['output_columns'])
    elif 'n_output_features' in manifest:
        n_out = manifest['n_output_features']

if n_out is not None and n_out <= 300 and hasattr(preprocessor, 'set_output'):
    preprocessor.set_output(transform='pandas')

k = min(20, n_out) if n_out is not None else min(20, len(feature_cols))
selector = SelectKBest(mutual_info_classif, k=k)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps) + [('selector', selector)]
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

mlca_runtime.fit_record(pipeline, X_train, partition='train', y=y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")

mlca_runtime.record_metric('n_selected_features', k, 'train')
mlca_runtime.record_metric('n_input_features', len(feature_cols), 'train')

print(f"Train shape: {train.shape}, Val shape: {val.shape if val is not None else 'None'}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Preprocessing output features: {n_out}")
print(f"Selected features (k): {k}")
