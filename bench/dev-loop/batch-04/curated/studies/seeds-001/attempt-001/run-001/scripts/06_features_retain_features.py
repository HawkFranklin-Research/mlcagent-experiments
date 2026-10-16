import mlca_runtime
import json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing} in _protocol. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

train = pd.read_parquet(inputs['train'])
drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == 'object' or str(s.dtype).startswith('string') or str(s.dtype).startswith('category'):
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

X = train[feature_cols]
selector = FunctionTransformer()
selector.set_output(transform='pandas')
mlca_runtime.fit_record(selector, X, partition='train')
mlca_runtime.save_model(selector, 'selector_all.pkl')
print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Selector fitted on {X.shape[1]} features")
print("Saved selector_all.pkl")
