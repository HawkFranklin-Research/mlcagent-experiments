import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = set(protocol.get('targets', []))
excluded = set(protocol.get('excluded_columns', []))
group = protocol.get('group_column', None)
drop_cols = targets | excluded | ({group} if group else set())

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_SYRRSV3B/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
else:
    candidates = [c for c in train.columns if c not in drop_cols]
    feature_cols = []
    for c in candidates:
        s = train[c]
        name = str(c).lower()
        if 'id' in name or 'index' in name or name == 'unnamed: 0':
            continue
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
            continue
        feature_cols.append(c)

bad = set(feature_cols) & drop_cols
if bad:
    raise ValueError(f"Feature columns contain target/excluded/group columns: {bad}")

X_train = train.loc[:, feature_cols].copy()

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
    step_name = 'selector_all'
    existing = set(preprocessor.named_steps.keys())
    while step_name in existing:
        step_name += '_'
    steps.append((step_name, 'passthrough'))
    selector_pipeline = Pipeline(steps)
else:
    selector_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector_all', 'passthrough'),
    ])

selector_pipeline = mlca_runtime.fit_record(selector_pipeline, X_train, 'train', y=None)
mlca_runtime.save_model(selector_pipeline, 'selector_all.pkl')

print(f"train rows={len(X_train)} features={len(feature_cols)}")
print("saved selector_all.pkl")
