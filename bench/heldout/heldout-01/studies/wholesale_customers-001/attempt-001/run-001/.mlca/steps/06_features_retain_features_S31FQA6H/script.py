import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_Q63KMMVN/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    raise ValueError("Preprocessor lacks feature_names_in_; cannot determine feature columns.")

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
forbidden = set(targets + excluded + ([group] if group else []))
intersect = set(feature_cols) & forbidden
if intersect:
    raise ValueError(f"Forbidden columns in feature set: {intersect}")

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns in train: {missing}")

X_train = train[feature_cols]

selector_all = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', 'passthrough')
])

selector_all = mlca_runtime.fit_record(selector_all, X_train, partition='train')
mlca_runtime.save_model(selector_all, 'selector_all.pkl')

print(f"Train shape: {train.shape}")
print(f"Feature count: {len(feature_cols)}")
print("Saved selector_all.pkl")
