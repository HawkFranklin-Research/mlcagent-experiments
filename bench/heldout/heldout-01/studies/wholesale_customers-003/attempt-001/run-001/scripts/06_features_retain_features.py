import json
import pandas as pd
import mlca_runtime
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

needed = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol['group_column']

train = pd.read_parquet(inputs['train'])


def is_identifier(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=True) == n
    return False


drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = [c for c in train.columns if c not in drop and not is_identifier(train[c])]
X = train.loc[:, feature_cols].copy()
X.index = train.index

selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X, 'train')
mlca_runtime.save_model(selector, 'selector_all.pkl')

print(f"train rows: {len(X)}")
print(f"feature cols: {len(feature_cols)}")
print(f"feature names: {feature_cols}")
print("saved selector_all.pkl")
