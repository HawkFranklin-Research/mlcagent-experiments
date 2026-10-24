import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_D50P13FR/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    feat_cols = list(pre.feature_names_in_)
else:
    feat_cols = [c for c in train.columns
                 if c not in targets and c not in excluded and c != group]

X = train[feat_cols]
y = train[targets[0]]

Xt = pre.transform(X)
n_feat = Xt.shape[1]
k = min(20, n_feat)

selector = SelectKBest(mutual_info_classif, k=k)
if isinstance(pre, Pipeline):
    full = Pipeline(list(pre.steps) + [('selector', selector)])
else:
    full = Pipeline([('preprocessor', pre), ('selector', selector)])

fitted = mlca_runtime.fit_record(full, X, 'train', y)
mlca_runtime.save_model(fitted, 'selector.pkl')
print(f"feat_cols={len(feat_cols)} encoded={n_feat} k={k}")
