import json, pickle
from functools import partial
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
try:
    targets = protocol['targets']
    excluded = set(protocol['excluded_columns'])
except KeyError as e:
    raise KeyError(f"Missing {e} in protocol; available keys: {list(protocol)}") from e
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_Y84CNP1M/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    src_cols = list(pre.feature_names_in_)
else:
    src_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group]

X = train[src_cols]
y = train[targets[0]].values

n_feat = pre.transform(X).shape[1]
k = min(20, n_feat)

score_func = partial(mutual_info_classif, random_state=42)
pipe = Pipeline([
    ('preprocessor', pre),
    ('selector', SelectKBest(score_func=score_func, k=k)),
])
mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(pipe, 'selector.pkl')
print('train rows', len(X), 'features_in', X.shape[1], 'k', k)
