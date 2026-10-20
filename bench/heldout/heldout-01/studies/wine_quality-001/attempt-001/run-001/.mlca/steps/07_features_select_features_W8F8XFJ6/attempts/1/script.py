import json, pickle
from functools import partial
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group]

with open(inputs['06_preprocess_fit_preprocessing_1DA5F11K/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

X_train = train[feature_cols]
y_train = train[targets[0]]
n_out = int(pre.transform(X_train.iloc[:5]).shape[1])
k = min(20, n_out)

pipe = Pipeline([
    ('pre', pre),
    ('select', SelectKBest(
        score_func=partial(mutual_info_classif, random_state=42),
        k=k,
    )),
])
mlca_runtime.fit_record(pipe, X_train, partition='train', y=y_train)

n_sel = int(pipe.named_steps['select'].get_support().sum())
mlca_runtime.record_metric('n_features_selected', float(n_sel), 'train')
mlca_runtime.save_model(pipe, 'selector.pkl')

print(f'features_in={len(feature_cols)} train_rows={len(train)}')
print(f'pre_out={n_out} k={k} selected={n_sel}')
print('saved selector.pkl')
