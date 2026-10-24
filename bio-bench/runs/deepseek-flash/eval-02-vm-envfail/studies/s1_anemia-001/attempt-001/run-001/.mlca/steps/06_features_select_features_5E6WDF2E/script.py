import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
y = train[targets[0]]

with open(inputs['05_preprocess_fit_preprocessing_HCSJFKGG/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    cols = [c for c in pre.feature_names_in_ if c not in targets and c not in excluded and c != group]
else:
    cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
X = train[cols].copy()

mlca_runtime.fit_record(pre, X, 'train', y)
n_out = pre.transform(X.iloc[:10]).shape[1]
k = min(20, n_out)

pipe = Pipeline([('pre', pre), ('sel', SelectKBest(mutual_info_classif, k=k))])
mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(pipe, 'selector.pkl')
print(f"features_in={X.shape[1]} features_out={n_out} k={k}")
