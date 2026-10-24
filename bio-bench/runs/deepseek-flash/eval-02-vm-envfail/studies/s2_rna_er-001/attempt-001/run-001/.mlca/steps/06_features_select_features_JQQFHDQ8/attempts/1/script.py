import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs['_protocol'].read_text())
targets = proto.get('targets') if proto.get('targets') is not None else proto['task']['targets']
excluded = set(proto['excluded_columns'])
group_col = proto.get('group_column')

train = pd.read_parquet(inputs['train'])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group_col is not None and c == group_col):
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith('string'):
        if s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

with open(inputs['05_preprocess_fit_preprocessing_R5N312M1/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    input_cols = [c for c in pre.feature_names_in_ if c in train.columns]
else:
    input_cols = feature_cols

bad = set(input_cols) & (set(targets) | excluded)
if bad:
    raise ValueError(f"Excluded/target columns reached preprocessor: {bad}")

y = train[targets[0]] if len(targets) == 1 else train[targets]
X = train[input_cols]

head = X.head(3) if len(X) >= 3 else X
n_out = pre.transform(head).shape[1]
k = min(20, n_out)

pipe = Pipeline([
    ('preprocessor', pre),
    ('selector', SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'selector.pkl')

sel = int(pipe.named_steps['selector'].get_support().sum())
mlca_runtime.record_metric('n_input_features', len(input_cols), 'train')
mlca_runtime.record_metric('n_preprocessed_features', int(n_out), 'train')
mlca_runtime.record_metric('n_selected_features', sel, 'train')
print(f"train rows {len(train)}; input features {len(input_cols)}; preprocessed {n_out}; selected {sel}")
