import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_CQ5BZKTY/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    manifest = json.loads(inputs['05_preprocess_fit_preprocessing_CQ5BZKTY/manifest.json'].read_text())
    for key in ['feature_columns', 'features', 'columns']:
        if key in manifest:
            feature_cols = manifest[key]
            break
    else:
        raise KeyError(f"Feature columns not found. Manifest keys: {list(manifest.keys())}")

for col in feature_cols:
    if col in targets or col in excluded or col == group_col:
        raise ValueError(f"Feature column {col} is a target/excluded/group column")

X_train = train[feature_cols].copy()
y_train = train[targets[0]]

if hasattr(preprocessor, 'get_feature_names_out'):
    try:
        n_features = len(preprocessor.get_feature_names_out())
    except Exception:
        mlca_runtime.fit_record(preprocessor, X_train, 'train')
        n_features = len(preprocessor.get_feature_names_out())
else:
    mlca_runtime.fit_record(preprocessor, X_train, 'train')
    n_features = preprocessor.transform(X_train).shape[1]

k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', selector)
])

mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)
mlca_runtime.save_model(pipeline, 'selector.pkl')
mlca_runtime.record_metric('n_features_preprocessed', n_features, 'train')
mlca_runtime.record_metric('n_features_selected', k, 'train')

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Preprocessed features: {n_features}")
print(f"Selected k: {k}")
print("Saved selector.pkl")
