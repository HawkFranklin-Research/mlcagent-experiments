import json
import pickle
from functools import partial

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_R1SQG5ET/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['05_preprocess_fit_preprocessing_R1SQG5ET/manifest.json'], 'r') as f:
    manifest = json.load(f)


def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing '{key}'. Available keys: {list(d.keys())}")
    return d[key]


targets = get_key(protocol, 'targets')
excluded = get_key(protocol, 'excluded_columns')
group_col = get_key(protocol, 'group_column')

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)


def key_like(s, name):
    if s.nunique(dropna=False) == len(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_integer_dtype(s):
            low = name.lower()
            if any(h in low for h in ['id', 'key', 'index', 'code']):
                return True
    return False


features = [c for c in train.columns if c not in drop_cols]
features = [c for c in features if not key_like(train[c], c)]

expected = None
if hasattr(preprocessor, 'feature_names_in_'):
    expected = list(preprocessor.feature_names_in_)
elif hasattr(preprocessor, 'named_steps'):
    for step in preprocessor.named_steps.values():
        if hasattr(step, 'feature_names_in_'):
            expected = list(step.feature_names_in_)
            break

if expected is not None:
    for c in expected:
        if c in drop_cols:
            raise ValueError(f"Preprocessor expects excluded/target column: {c}")
    features = expected

X_train = train[features]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

X_trans = preprocessor.transform(X_train.iloc[:1])
n_features = X_trans.shape[1]
k = min(20, n_features)

selector = SelectKBest(
    score_func=partial(mutual_info_classif, random_state=42),
    k=k,
)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
else:
    steps = [('preprocess', preprocessor)]
steps.append(('select', selector))
pipeline = Pipeline(steps)

pipeline = mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")

print(f"Train/val shapes: {train.shape}, {val.shape}")
print(f"Features: {len(features)}, preprocessed: {n_features}, k={k}")
