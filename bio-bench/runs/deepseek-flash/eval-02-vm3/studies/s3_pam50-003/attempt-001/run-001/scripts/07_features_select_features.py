import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_6C1GR31N/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = [c for c in preprocessor.feature_names_in_ if c not in targets]
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = [c for c in train.columns if c not in drop]

X = train[feature_cols]
y = train[targets[0]]

Xt = preprocessor.transform(X)
n_feat = Xt.shape[1]
k = min(20, int(n_feat))

selector = Pipeline([
    ('pre', preprocessor),
    ('sel', SelectKBest(mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(selector, X, 'train', y=y)
mlca_runtime.save_model(selector, 'selector.pkl')
print(f"features_in={len(feature_cols)} post_pre={n_feat} k={k}")
