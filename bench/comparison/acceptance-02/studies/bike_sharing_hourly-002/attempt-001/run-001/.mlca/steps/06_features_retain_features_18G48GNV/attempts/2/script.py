import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for k in ('targets', 'excluded_columns', 'task'):
    if k not in protocol:
        raise KeyError(f"Missing key {k}; available: {list(protocol.keys())}")

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_7BD2DR19/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    raw_names = list(preprocessor.feature_names_in_)
else:
    raw_names = list(train.columns)

feature_cols = [c for c in raw_names if c in train.columns and c not in drop]
if len(feature_cols) < 3:
    feature_cols = [c for c in train.columns if c not in drop]
    n = len(train)
    keep = []
    for c in feature_cols:
        s = train[c]
        if s.dtype == 'object' and s.nunique() == n:
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique() == n:
            continue
        keep.append(c)
    feature_cols = keep

X_raw = train[feature_cols]
X_pre = preprocessor.transform(X_raw)

selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X_pre, partition='train')

pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
mlca_runtime.save_model(pipeline, 'selector_all.pkl')

n_in = len(feature_cols)
n_out = getattr(X_pre, 'shape', (len(X_raw), '?'))[1] if hasattr(X_pre, 'shape') else '?'
print(f"In: {n_in}  Out: {n_out}")
print("Saved selector_all.pkl")
