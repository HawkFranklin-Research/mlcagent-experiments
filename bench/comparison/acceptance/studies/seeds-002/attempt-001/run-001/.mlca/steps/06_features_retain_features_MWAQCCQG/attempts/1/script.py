import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise ValueError(f"Missing key {k} in _protocol. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
all_cols = train.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and (group_col is None or c != group_col)]
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            feature_cols.remove(c)

with open(inputs['05_preprocess_fit_preprocessing_XP5A8XH9/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)

if isinstance(preproc, Pipeline):
    base_steps = preproc.steps
    first_step = preproc.steps[0][1]
    input_cols = first_step.feature_names_in_.tolist() if hasattr(first_step, 'feature_names_in_') else feature_cols
else:
    base_steps = [('preprocessor', preproc)]
    input_cols = preproc.feature_names_in_.tolist() if hasattr(preproc, 'feature_names_in_') else feature_cols

if any(c in targets or c in excluded or (group_col is not None and c == group_col) for c in input_cols):
    raise ValueError("Preprocessor input contains target, excluded, or group column.")

X_raw = train[input_cols]
X_pre = preproc.transform(X_raw)
if not isinstance(X_pre, pd.DataFrame):
    if hasattr(preproc, 'get_feature_names_out'):
        X_pre = pd.DataFrame(X_pre, columns=preproc.get_feature_names_out(), index=train.index)
    elif X_pre.shape[1] == len(input_cols):
        X_pre = pd.DataFrame(X_pre, columns=input_cols, index=train.index)
    else:
        raise ValueError("Cannot determine feature names for preprocessed data")

selector_cols = X_pre.columns.tolist()
selector = ColumnTransformer([('passthrough', 'passthrough', selector_cols)], remainder='drop')
if len(selector_cols) <= 300:
    selector.set_output(transform='pandas')

selector = mlca_runtime.fit_record(selector, X_pre, partition='train')

final_steps = base_steps + [('selector_all', selector)]
final_pipeline = Pipeline(final_steps)
if len(selector_cols) <= 300:
    final_pipeline.set_output(transform='pandas')

mlca_runtime.save_model(final_pipeline, 'selector_all.pkl')

print(f"Raw feature columns: {len(feature_cols)}")
print(f"Preprocessed features: {len(selector_cols)}")
print(f"Selected features: {len(selector_cols)}")
print(f"Training rows: {len(X_raw)}")
print("Saved selector_all.pkl")
