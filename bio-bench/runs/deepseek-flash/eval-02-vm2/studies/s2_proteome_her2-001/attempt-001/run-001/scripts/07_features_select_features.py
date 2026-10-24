import mlca_runtime
import json
import pickle
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import clone

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_R9QMSZCE/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = [c for c in preprocessor.feature_names_in_ if c in train.columns]
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group]

X_train = train[feature_cols]
y_train = train[targets[0]]

pre_check = clone(preprocessor)
mlca_runtime.fit_record(pre_check, X_train, 'train')
try:
    n_out = pre_check.transform(X_train).shape[1]
except Exception:
    n_out = len(feature_cols)
k = min(20, n_out)
print(f"features {len(feature_cols)}, width {n_out}, k={k}")

score = partial(mutual_info_classif, random_state=42)
if isinstance(preprocessor, Pipeline):
    pipeline = Pipeline(list(preprocessor.steps) + [('select', SelectKBest(score_func=score, k=k))])
else:
    pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('select', SelectKBest(score_func=score, k=k)),
    ])
mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipeline, 'selector.pkl')
print("saved selector.pkl")
