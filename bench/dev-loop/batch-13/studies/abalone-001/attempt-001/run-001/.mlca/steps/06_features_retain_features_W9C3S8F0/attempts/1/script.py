import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing required protocol key '{k}'. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

df = pd.read_parquet(inputs['train'])
all_cols = list(df.columns)
candidate_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

feature_cols = []
for c in candidate_cols:
    col = df[c]
    if col.dtype == 'object' and col.nunique() == len(df):
        continue
    if col.dtype.kind in 'iu' and col.nunique() == len(df):
        continue
    feature_cols.append(c)

X = df[feature_cols]

selector = ColumnTransformer(
    transformers=[('passthrough', 'passthrough', feature_cols)],
    remainder='drop'
)
selector.set_output(transform='pandas')

mlca_runtime.fit_record(selector, X, 'train')
mlca_runtime.save_model(selector, 'selector_all.pkl')

print(f"Selected {len(feature_cols)} features: {feature_cols}")
print(f"Training rows: {len(df)}")
