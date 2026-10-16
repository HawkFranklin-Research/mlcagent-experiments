import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

train = pd.read_parquet(inputs['train'])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'string':
        if s.nunique(dropna=True) == s.notna().sum():
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == s.notna().sum():
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

X_train = train[feature_cols]

with open(inputs['05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
else:
    steps = [('preprocessor', preprocessor)]

pipe = Pipeline(steps + [('selector', FunctionTransformer())])

mlca_runtime.fit_record(pipe, X_train, 'train')

mlca_runtime.save_model(pipe, 'selector_all.pkl')

print(f"features: {len(feature_cols)}")
print(f"train rows: {len(X_train)}")
