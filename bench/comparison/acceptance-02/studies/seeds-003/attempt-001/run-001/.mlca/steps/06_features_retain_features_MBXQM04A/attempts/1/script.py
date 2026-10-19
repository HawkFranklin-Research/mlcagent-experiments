import json
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import VarianceThreshold

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol.keys())}")

df = pd.read_parquet(inputs['train'])
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

def key_like(s):
    kind = s.dtype.kind
    return (kind in 'iu' or s.dtype == object) and s.nunique(dropna=False) == len(s)

feature_cols = [c for c in df.columns if c not in drop and not key_like(df[c])]
X = df[feature_cols]

selector = VarianceThreshold(threshold=-1.0).set_output(transform='pandas')
selector = mlca_runtime.fit_record(selector, X, 'train')
mlca_runtime.save_model(selector, 'selector_all.pkl')
mlca_runtime.record_metric('n_features', float(len(feature_cols)), 'train')
print(f"Selector fit on {X.shape[0]} rows, {X.shape[1]} features")
