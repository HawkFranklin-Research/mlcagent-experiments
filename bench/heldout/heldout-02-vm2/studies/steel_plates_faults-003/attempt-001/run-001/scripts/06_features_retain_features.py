import mlca_runtime
import json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column']:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

train = pd.read_parquet(inputs['train'])
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train) and train[c].notna().all():
            continue
    feature_cols.append(c)

X = train[feature_cols]
selector = FunctionTransformer(func=None, feature_names_out='one-to-one', validate=False)
selector = mlca_runtime.fit_record(selector, X, partition='train')
mlca_runtime.save_model(selector, 'selector_all.pkl')
print(f"Fitted passthrough selector on {X.shape[0]} rows, {X.shape[1]} features.")
print(f"Feature columns: {feature_cols[:5]}{' ...' if len(feature_cols) > 5 else ''}")
