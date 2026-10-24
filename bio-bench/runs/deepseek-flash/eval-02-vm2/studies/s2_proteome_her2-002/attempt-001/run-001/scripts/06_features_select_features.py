import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_F5B5FT5N/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    feature_cols = [c for c in train.columns
                    if c not in targets
                    and c not in excluded
                    and c != group_col]

X = train[feature_cols]
y = train[targets[0]]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    n_features = preprocessor.transform(X.head(5)).shape[1]

k = min(20, n_features)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('select', SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, 'selector.pkl')

print(f"Train: {train.shape[0]} rows, {len(feature_cols)} features")
print(f"Selected k={k} of {n_features}")
