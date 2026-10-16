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
    raise ValueError(f"Missing keys {missing} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

train = pd.read_parquet(inputs['train'])

all_cols = train.columns.tolist()
candidate_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

def is_key_like(series):
    if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
        return series.nunique(dropna=False) == len(series)
    if pd.api.types.is_integer_dtype(series):
        if series.nunique(dropna=False) == len(series):
            vals = series.dropna().sort_values()
            if len(vals) > 0 and (vals.max() - vals.min() + 1) == len(vals):
                return True
    return False

feature_cols = [c for c in candidate_cols if not is_key_like(train[c])]

X = train[feature_cols].copy()

selector = FunctionTransformer(func=None, validate=False, feature_names_out='one-to-one')
mlca_runtime.fit_record(selector, X, partition='train')

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train shape: {train.shape}")
print(f"feature count: {len(feature_cols)}")
print(f"features: {feature_cols}")
