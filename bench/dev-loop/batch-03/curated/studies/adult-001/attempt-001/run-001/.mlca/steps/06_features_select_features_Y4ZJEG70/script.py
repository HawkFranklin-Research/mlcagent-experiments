import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns'] or []
group = protocol['group_column']

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_F5B2KN9D/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

def get_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

feature_cols = get_input_cols(preprocessor)
if feature_cols is None:
    try:
        with open(inputs['05_preprocess_fit_preprocessing_F5B2KN9D/manifest.json']) as f:
            manifest = json.load(f)
        for key in ['feature_columns', 'features', 'input_columns']:
            if key in manifest:
                feature_cols = manifest[key]
                break
    except Exception:
        pass

if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object or s.dtype.name.startswith('string'):
            if s.nunique(dropna=False) == len(s):
                continue
        if pd.api.types.is_integer_dtype(s):
            if s.nunique(dropna=False) == len(s) and ('id' in c.lower() or 'code' in c.lower()):
                continue
        feature_cols.append(c)

for c in set(targets) | set(excluded):
    if c in feature_cols:
        raise ValueError(f"Column '{c}' is a target/excluded but present in feature columns")

X_train = train[feature_cols]
y_train = train[targets[0]]

X_trans = preprocessor.transform(X_train)
n_features = X_trans.shape[1]
k = min(20, n_features)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps) + [('selector', SelectKBest(score_func=mutual_info_classif, k=k))]
    selector_pipeline = Pipeline(steps)
else:
    selector_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_classif, k=k))
    ])

mlca_runtime.fit_record(selector_pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(selector_pipeline, 'selector.pkl')

print(f"Train rows: {len(X_train)}")
print(f"Input features: {len(feature_cols)}")
print(f"Preprocessed features: {n_features}")
print(f"Selected k: {k}")
