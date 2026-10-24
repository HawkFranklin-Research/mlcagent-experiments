import mlca_runtime
import json
import pickle
from functools import partial
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import clone

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
with open(inputs['06_preprocess_fit_preprocessing_R9QMSZCE/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    feature_cols = []
    for c in cols:
        s = train[c]
        if s.dtype == 'object' or s.dtype.name == 'string':
            if s.nunique() == len(s):
                continue
        feature_cols.append(c)

X_train = train[feature_cols]
y_train = train[targets[0]]

pre_clone = clone(preprocessor)
pre_clone.fit(X_train)
n_out = pre_clone.transform(X_train[:5]).shape[1]
k = min(20, n_out)
print(f"train shape {train.shape}, features {len(feature_cols)}, preprocessed width {n_out}, k={k}")

pipeline = Pipeline([
    ('preprocessor', pre_clone),
    ('select', SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k))
])

mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipeline, 'selector.pkl')
print("saved selector.pkl")
