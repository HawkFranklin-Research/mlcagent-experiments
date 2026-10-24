import json, pickle
from functools import partial
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_AP3P14P9/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    feat_cols = list(pre.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feat_cols = [c for c in train.columns if c not in drop]

X_train = train[feat_cols]
y_train = train[target]

Xt = pre.transform(X_train)
n_out = int(Xt.shape[1])
k = min(20, n_out)

pipe = Pipeline([
    ('preprocessor', pre),
    ('selector', SelectKBest(
        score_func=partial(mutual_info_classif, random_state=0),
        k=k,
    )),
])

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(fitted, 'selector.pkl')

mlca_runtime.record_metric('n_input_features', float(len(feat_cols)), 'train')
mlca_runtime.record_metric('n_transformed_features', float(n_out), 'train')
mlca_runtime.record_metric('n_selected_features', float(k), 'train')
print(f'input={len(feat_cols)} transformed={n_out} selected={k}')
