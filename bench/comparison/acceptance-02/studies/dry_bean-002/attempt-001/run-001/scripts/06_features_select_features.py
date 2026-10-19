import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_MQJ4AQ8S/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

all_cols = list(train.columns)
feat_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            return True
    if pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s) and not s.isin([0, 1]).all():
            return True
    return False

feat_cols = [c for c in feat_cols if not is_key_like(train[c])]
X_train = train[feat_cols]
y_train = train[targets[0]]

sample = X_train.head(50)
try:
    width = preprocessor.transform(sample).shape[1]
except Exception:
    width = len(feat_cols)

k = min(20, width)
pipeline = Pipeline([
    ('pre', preprocessor),
    ('select', SelectKBest(mutual_info_classif, k=k))
])

mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)
mlca_runtime.save_model(pipeline, 'selector.pkl')
print(f"Saved selector.pkl (k={k}, n_features={width})")
