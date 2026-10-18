import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ('targets', 'excluded_columns', 'group_column', 'task'):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

train_df = pd.read_parquet(inputs['train'])

feature_cols = []
for col in train_df.columns:
    if col in targets or col in excluded or col == group:
        continue
    s = train_df[col]
    is_text = s.dtype == 'object' or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(col)

selector = ColumnTransformer(
    [('keep', 'passthrough', feature_cols)],
    remainder='drop'
).set_output(transform='pandas')

selector_all = mlca_runtime.fit_record(selector, train_df, 'train')

mlca_runtime.save_model(selector_all, 'selector_all.pkl')

print(f"train shape {train_df.shape}")
print(f"selected features {len(feature_cols)}")
print(f"input columns {list(selector_all.feature_names_in_)}")
