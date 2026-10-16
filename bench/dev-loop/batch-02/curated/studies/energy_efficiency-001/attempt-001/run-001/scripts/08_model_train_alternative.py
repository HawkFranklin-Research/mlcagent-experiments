import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_8F2KVPF9/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

feature_cols = None
if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
elif hasattr(selector, 'feature_names_in_'):
    feature_cols = list(selector.feature_names_in_)
if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
    key_like = []
    for c in feature_cols:
        s = train[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            key_like.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
            if len(s) > 1 and (s.diff().dropna() == 1).all():
                key_like.append(c)
    feature_cols = [c for c in feature_cols if c not in key_like]

feature_cols = [c for c in feature_cols if c not in set(targets) | set(excluded) | ({group} if group else set())]

use_selector_as_front = False
if isinstance(selector, Pipeline) and 'preprocessor' in selector.named_steps:
    use_selector_as_front = True
elif hasattr(selector, 'feature_names_in_') and set(selector.feature_names_in_) == set(feature_cols):
    use_selector_as_front = True

if use_selector_as_front:
    pipeline = Pipeline([('front', selector), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
else:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

X = train[feature_cols]
y = train[targets]

pipeline = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, 'alternative_model.pkl')
print(f"train shape: {train.shape}, features: {len(feature_cols)}, targets: {len(targets)}")
