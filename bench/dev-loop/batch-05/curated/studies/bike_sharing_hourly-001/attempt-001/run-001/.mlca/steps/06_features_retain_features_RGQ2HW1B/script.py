import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
all_cols = list(train.columns)
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
for c in all_cols:
    if c in drop:
        continue
    if train[c].dtype == 'object' or pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            drop.add(c)
feature_cols = [c for c in all_cols if c not in drop]
X_train = train[feature_cols]

with open(inputs['05_preprocess_fit_preprocessing_VM7ERF98/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

selector = FunctionTransformer(func=None, feature_names_out='one-to-one')
if isinstance(pre, Pipeline):
    model = Pipeline(list(pre.steps) + [('selector', selector)])
else:
    model = Pipeline([('preprocessor', pre), ('selector', selector)])

mlca_runtime.fit_record(model, X_train, 'train')
mlca_runtime.save_model(model, "selector_all.pkl")
print(f"X_train shape: {X_train.shape}")
print(f"Features: {len(feature_cols)}")
